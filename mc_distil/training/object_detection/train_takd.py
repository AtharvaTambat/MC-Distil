import os
import torch
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from PIL import Image
from argparse import ArgumentParser
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Data Handling Section
class ToTensor:
    def __call__(self, img, target):
        return torchvision.transforms.functional.to_tensor(img), target

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if torch.rand(1) < self.p:
            img = torchvision.transforms.functional.hflip(img)
            if 'boxes' in target and target['boxes'].shape[0] > 0:
                img_width = img.shape[2]
                target['boxes'][:, [0, 2]] = img_width - target['boxes'][:, [2, 0]]
        return img, target

class CustomCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root, self.transforms, self.coco = root, transforms, COCO(annotation_file)
        self.ids = [i for i in sorted(self.coco.getImgIds()) if len(self.coco.getAnnIds(imgIds=i, iscrowd=False)) > 0]

    def __getitem__(self, index):
        img_id = self.ids[index]
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        boxes, labels = [], []
        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes.append([xmin, ymin, xmin + w, ymin + h])
                labels.append(ann['category_id'])
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,)),
            "image_id": torch.tensor([img_id])
        }
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)

class COCODataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, workers):
        super().__init__()
        self.data_path, self.batch_size, self.workers = data_path, batch_size, workers

    def setup(self, stage=None):
        train_dir, train_ann = os.path.join(self.data_path, 'train2017'), os.path.join(self.data_path, 'annotations/instances_train2017.json')
        val_dir, val_ann = os.path.join(self.data_path, 'val2017'), os.path.join(self.data_path, 'annotations/instances_val2017.json')
        self.train_dataset = COCODataset(train_dir, train_ann, self._get_transforms(True))
        self.val_dataset = COCODataset(val_dir, val_ann, self._get_transforms(False))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.workers, pin_memory=True, collate_fn=lambda b: tuple(zip(*b)))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=True, collate_fn=lambda b: tuple(zip(*b)))

    def _get_transforms(self, is_train):
        return CustomCompose([ToTensor(), RandomHorizontalFlip()]) if is_train else CustomCompose([ToTensor()])

# Reusable Callbacks
class LearningRateWarmUp(pl.Callback):
    def __init__(self, warmup_steps, target_lr, initial_lr=1e-6):
        super().__init__()
        self.warmup_steps, self.target_lr, self.initial_lr = warmup_steps, target_lr, initial_lr

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, (trainer.global_step + 1) / self.warmup_steps)
            for pg in trainer.optimizers[0].param_groups:
                pg['lr'] = self.initial_lr + lr_scale * (self.target_lr - self.initial_lr)

# Lightning Module for Single-Step KD
class SingleStepKDLitModule(pl.LightningModule):
    def __init__(self, student_backbone, teacher_backbone, student_weights_path, teacher_weights_path, lr, weight_decay, alpha_kd, temperature, num_classes=91):
        super().__init__()
        self.save_hyperparameters()
        self.student = self._create_and_load_model(student_backbone, num_classes, student_weights_path, "Student")
        self.teacher = self._create_and_load_model(teacher_backbone, num_classes, teacher_weights_path, "Teacher")

        # Get the number of output channels from the FPN of each model's backbone
        student_out_channels = self.student.backbone.out_channels
        teacher_out_channels = self.teacher.backbone.out_channels
        
        # If the channel counts are the same, no adaptation is needed.
        if student_out_channels == teacher_out_channels:
            self.adaptation_layer = torch.nn.Identity()
            print("Channels match. Using nn.Identity() as the adaptation layer.")
        else:
            # If channels differ, create a 1x1 convolution to match them.
            # This layer will learn to adapt the student's features to the
            # teacher's feature space.
            self.adaptation_layer = torch.nn.Conv2d(
                in_channels=student_out_channels,
                out_channels=teacher_out_channels,
                kernel_size=1
            )
            print(f"Channels differ. Creating a 1x1 Conv2d adaptation layer.")

        self._freeze_model(self.teacher)
        self.val_map = MeanAveragePrecision(box_format='xyxy')
        print(f"\nKD Step Setup: Teacher='{teacher_backbone}', Student='{student_backbone}'\n")

    def _create_and_load_model(self, backbone, num_classes, ckpt_path, name):
        model = torchvision.models.detection.FasterRCNN(
            torchvision.models.detection.backbone_utils.resnet_fpn_backbone(backbone, pretrained=True),
            num_classes
        )
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        ckpt_dict = checkpoint.get('state_dict', checkpoint)
        final_model_dict = None
        prefixes_to_try = ['student.', 'teacher.', 'model.']
        for prefix in prefixes_to_try:
            temp_dict = {k.replace(prefix, ''): v for k, v in ckpt_dict.items() if k.startswith(prefix)}
            if temp_dict:
                final_model_dict = temp_dict
                print(f"Found and stripped prefix '{prefix}' from checkpoint keys for {name}.")
                break
        if final_model_dict is None:
            final_model_dict = ckpt_dict
            print(f"Loading checkpoint with standard keys (no prefix found) for {name}.")
        model.load_state_dict(final_model_dict, strict=True)
        print(f"Successfully loaded {name} weights from {ckpt_path}")
        return model

    @staticmethod
    def _freeze_model(model):
        model.eval()
        [p.requires_grad_(False) for p in model.parameters()]

    def forward(self, images, targets=None):
        return self.student(images, targets)

    def _get_cls_logits(self, model, features, proposals, image_sizes):
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_sizes)
        return model.roi_heads.box_predictor(model.roi_heads.box_head(box_features))[0]
    
    def _get_reg_preds(self, model, features, proposals, image_sizes):
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_sizes)
        box_features = model.roi_heads.box_head(box_features)
        _, box_regression = model.roi_heads.box_predictor(box_features)
        return box_regression

    def _calculate_hint_loss(self, student_features, teacher_features):
        student_fmap = student_features['pool']
        teacher_fmap = teacher_features['pool']

        # Apply the adaptation layer to the student's feature map.
        # This will either be a 1x1 convolution or an Identity layer.
        adapted_student_fmap = self.adaptation_layer(student_fmap)

        # Calculate the Mean Squared Error (L2) loss between the feature maps.
        loss = torch.nn.functional.mse_loss(adapted_student_fmap, teacher_fmap)
        return loss

    def training_step(self, batch, batch_idx):
        images, targets = batch
        batch_size = len(images)
        images, targets = self.student.transform(images, targets)

        # Standard Student Object Detection Loss 
        student_features = self.student.backbone(images.tensors)
        proposals, p_loss = self.student.rpn(images, student_features, targets)
        _, d_loss = self.student.roi_heads(student_features, proposals, images.image_sizes, targets)
        loss_obj_detect = sum(loss for loss in {**p_loss, **d_loss}.values())

        # Knowledge Distillation Losses
        self.teacher.eval()
        with torch.no_grad():
            teacher_features = self.teacher.backbone(images.tensors)
            teacher_proposals, _ = self.teacher.rpn(images, teacher_features)
            
            # Get all necessary teacher predictions
            teacher_cls_logits = self._get_cls_logits(self.teacher, teacher_features, teacher_proposals, images.image_sizes)
            teacher_reg_preds = self._get_reg_preds(self.teacher, teacher_features, teacher_proposals, images.image_sizes)

        student_cls_logits = self._get_cls_logits(self.student, student_features, teacher_proposals, images.image_sizes)
        T = self.hparams.temperature
        loss_cls_kd = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_cls_logits / T, 1),
            torch.nn.functional.softmax(teacher_cls_logits / T, 1),
            reduction='batchmean'
        ) * (T**2)

        student_reg_preds = self._get_reg_preds(self.student, student_features, teacher_proposals, images.image_sizes)
        loss_reg_kd = torch.nn.functional.smooth_l1_loss(student_reg_preds, teacher_reg_preds)

        loss_hint = self._calculate_hint_loss(student_features, teacher_features)

        beta_reg = 1.0   # Weight for the regression distillation loss
        gamma_hint = 0.5 # Weight for the feature hint loss

        total_distillation_loss = (loss_cls_kd) + (beta_reg * loss_reg_kd) + (gamma_hint * loss_hint)
        loss = (1 - self.hparams.alpha_kd) * loss_obj_detect + self.hparams.alpha_kd * total_distillation_loss

        self.log_dict({
            'loss': loss,
            'loss_obj_detect': loss_obj_detect,
            'loss_kd_cls': loss_cls_kd,
            'loss_kd_reg': loss_reg_kd,
            'loss_hint': loss_hint,
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        return loss

    def validation_step(self, batch, batch_idx):
        self.val_map.update(self.student(batch[0]), batch[1])

    def on_validation_epoch_end(self):
        metrics = self.val_map.compute()
        self.log_dict({"val_AP": metrics["map"], "val_AP50": metrics["map_50"], "val_AP75": metrics["map_75"]}, prog_bar=True)
        self.val_map.reset()

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.student.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[8, 11], gamma=0.1)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

# Main Script
def main():
    parser = ArgumentParser(description="Single-Step Knowledge Distillation for Object Detection on the full COCO dataset.")
    parser.add_argument("--data_path", required=True, help="Path to MS COCO dataset root.")
    parser.add_argument("--output_dir", required=True, help="Directory to save checkpoints.")
    parser.add_argument("--student_weights_path", required=True, help="Path to pre-trained student .ckpt.")
    parser.add_argument("--teacher_weights_path", required=True, help="Path to pre-trained teacher .ckpt.")
    parser.add_argument("--student_backbone", required=True, help="Student backbone name (e.g., resnet18).")
    parser.add_argument("--teacher_backbone", required=True, help="Teacher backbone name (e.g., resnet34).")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU.")
    parser.add_argument("--lr", type=float, default=0.002, help="Target learning rate after warm-up.")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--alpha_kd", type=float, default=0.5, help="Weight for KD loss.")
    parser.add_argument("--temperature", type=float, default=4.0, help="Temperature for softening logits.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--gpus", type=str, default='0', help="GPU IDs to use.")
    args = parser.parse_args()

    pl.seed_everything(42, workers=True)
    coco_dm = COCODataModule(args.data_path, args.batch_size, args.workers)
    model = SingleStepKDLitModule(**vars(args))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_AP", mode="max", dirpath=args.output_dir, save_top_k=3, filename='{epoch:02d}-{val_AP:.3f}')
    callbacks = [checkpoint_callback, pl.callbacks.LearningRateMonitor('epoch'), LearningRateWarmUp(500, args.lr)]

    accelerator = "gpu"
    if ',' in args.gpus:
        devices, strategy = [int(i.strip()) for i in args.gpus.split(',')], "ddp"
    else:
        devices, strategy = int(args.gpus), "auto"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=args.epochs,
        precision="16-mixed",
        callbacks=callbacks,
        log_every_n_steps=20
    )
    trainer.fit(model, datamodule=coco_dm)
    print(f"\nTraining step finished. Best model checkpoint saved in: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
