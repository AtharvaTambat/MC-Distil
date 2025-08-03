import os
import torch
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from PIL import Image
from argparse import ArgumentParser
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class ToTensor(object):
    def __call__(self, img, target):
        return torchvision.transforms.functional.to_tensor(img), target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img, target):
        if torch.rand(1) < self.p:
            img = torchvision.transforms.functional.hflip(img)
            if 'boxes' in target and target['boxes'].shape[0] > 0:
                img_width = img.shape[2]; target['boxes'][:, [0, 2]] = img_width - target['boxes'][:, [2, 0]]
        return img, target

class CustomCompose(object):
    def __init__(self, transforms): self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms: image, target = t(image, target)
        return image, target

class COCODataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the MS COCO dataset."""
    def __init__(self, data_path: str, batch_size: int, workers: int):
        super().__init__()
        self.data_path, self.batch_size, self.workers = data_path, batch_size, workers
    
    def setup(self, stage=None):
        train_dir = os.path.join(self.data_path, 'train2017'); train_ann = os.path.join(self.data_path, 'annotations/instances_train2017.json')
        val_dir = os.path.join(self.data_path, 'val2017'); val_ann = os.path.join(self.data_path, 'annotations/instances_val2017.json')
        self.train_dataset = COCODataset(root=train_dir, annotation_file=train_ann, transforms=self._get_transforms(is_train=True))
        self.val_dataset = COCODataset(root=val_dir, annotation_file=val_ann, transforms=self._get_transforms(is_train=False))
    
    def train_dataloader(self): 
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, pin_memory=True, collate_fn=self.collate_fn)
    
    def val_dataloader(self): 
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=True, collate_fn=self.collate_fn)
   
    @staticmethod
    def collate_fn(batch): 
        return tuple(zip(*batch))
    
    @staticmethod
    def _get_transforms(is_train=True):
        transforms = [ToTensor()]
        if is_train: transforms.append(RandomHorizontalFlip())
        return CustomCompose(transforms)

class COCODataset(torch.utils.data.Dataset):
    """MS COCO Dataset for Object Detection."""
    def __init__(self, root, annotation_file, transforms=None):
        self.root, self.transforms, self.coco = root, transforms, COCO(annotation_file)
        ids = sorted(self.coco.getImgIds()); self.ids = [i for i in ids if len(self.coco.getAnnIds(imgIds=i, iscrowd=False)) > 0]
    
    def __getitem__(self, index):
        img_id = self.ids[index]; path = self.coco.loadImgs(img_id)[0]['file_name']; img = Image.open(os.path.join(self.root, path)).convert('RGB')
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False)); boxes, labels = [], []
        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            if w > 0 and h > 0: boxes.append([xmin, ymin, xmin + w, ymin + h]); labels.append(ann['category_id'])
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4)), "labels": torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64), "image_id": torch.tensor([img_id])}
        if self.transforms: img, target = self.transforms(img, target)
        return img, target
    
    def __len__(self): return len(self.ids)


class KD_FasterRCNN_Lightning(pl.LightningModule):
    def __init__(self, student_backbone: str, teacher_backbone: str, lr: float, weight_decay: float,
                 alpha_kd: float, temperature: float, student_weights_path: str = None,
                 teacher_weights_path: str = None, num_classes: int = 91):
        super().__init__()
        self.save_hyperparameters()
        self.student = self._create_model(student_backbone, num_classes)
        if self.hparams.student_weights_path:
            self._load_from_lightning_checkpoint(self.student, self.hparams.student_weights_path, "Student")
        self.teacher = self._create_model(teacher_backbone, num_classes)
        if self.hparams.teacher_weights_path:
            self._load_from_lightning_checkpoint(self.teacher, self.hparams.teacher_weights_path, "Teacher")
        else:
            print("Using default torchvision COCO-pretrained teacher.")
        self.teacher.eval()

        student_out_channels = self.student.backbone.out_channels
        teacher_out_channels = self.teacher.backbone.out_channels
        
        if student_out_channels == teacher_out_channels:
            self.adaptation_layer = torch.nn.Identity()
            print("Channels match. Using nn.Identity() as the adaptation layer.")
        else:
            self.adaptation_layer = torch.nn.Conv2d(
                in_channels=student_out_channels,
                out_channels=teacher_out_channels,
                kernel_size=1
            )
            print(f"Channels differ. Creating a 1x1 Conv2d adaptation layer.")

        self.val_map = MeanAveragePrecision(box_format='xyxy')

    @staticmethod
    def _load_from_lightning_checkpoint(model, ckpt_path, model_name="Model"):
        print(f"Loading {model_name} weights from Lightning checkpoint: {ckpt_path}")
        if not os.path.exists(ckpt_path):
            print(f"WARNING: {model_name} checkpoint file not found at {ckpt_path}. Model will not be loaded.")
            return
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        lightning_state_dict = checkpoint['state_dict']
        new_state_dict = {}
        prefix = 'model.'
        for k, v in lightning_state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                new_state_dict[new_key] = v
        if not new_state_dict:
            raise KeyError(f"Could not find any keys with the prefix '{prefix}' in the checkpoint.")
        model.load_state_dict(new_state_dict)
        print(f"{model_name} weights loaded successfully.")

    def _create_model(self, backbone_name, num_classes):
        print(f"Building Faster R-CNN with {backbone_name} backbone...")
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(backbone_name, pretrained=True)
        model = torchvision.models.detection.FasterRCNN(backbone, num_classes)
        return model

    def forward(self, images, targets=None):
        return self.student(images, targets)

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
            teacher_cls_logits = self._get_cls_logits(self.teacher, teacher_features, teacher_proposals, images.image_sizes)
            teacher_reg_preds = self._get_reg_preds(self.teacher, teacher_features, teacher_proposals, images.image_sizes)

        # Classification Distillation
        student_cls_logits = self._get_cls_logits(self.student, student_features, teacher_proposals, images.image_sizes)
        T = self.hparams.temperature
        loss_cls_kd = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_cls_logits / T, dim=1),
            torch.nn.functional.softmax(teacher_cls_logits / T, dim=1),
            reduction='batchmean'
        ) * (T**2)

        # Regression Distillation
        student_reg_preds = self._get_reg_preds(self.student, student_features, teacher_proposals, images.image_sizes)
        loss_reg_kd = torch.nn.functional.smooth_l1_loss(student_reg_preds, teacher_reg_preds)

        # Feature Hint Distillation
        loss_hint = self._calculate_hint_loss(student_features, teacher_features)

        beta_reg = 1.0   # Weight for the regression distillation loss
        gamma_hint = 0.5 # Weight for the feature hint loss
        total_distillation_loss = (loss_cls_kd) + (beta_reg * loss_reg_kd) + (gamma_hint * loss_hint)        
        loss = (1 - self.hparams.alpha_kd) * loss_obj_detect + self.hparams.alpha_kd * total_distillation_loss

        self.log_dict({
            'train/loss': loss,
            'train/loss_obj_detect': loss_obj_detect,
            'train/loss_kd_cls': loss_cls_kd,
            'train/loss_kd_reg': loss_reg_kd,
            'train/loss_hint': loss_hint,
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        return loss

    def _get_cls_logits(self, model, features, proposals, image_sizes):
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_sizes)
        box_features = model.roi_heads.box_head(box_features)
        cls_logits, _ = model.roi_heads.box_predictor(box_features)
        return cls_logits

    def _get_reg_preds(self, model, features, proposals, image_sizes):
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_sizes)
        box_features = model.roi_heads.box_head(box_features)
        _, box_regression = model.roi_heads.box_predictor(box_features)
        return box_regression

    def _calculate_hint_loss(self, student_features, teacher_features):
        student_fmap = student_features.get('pool', student_features.get('0'))
        teacher_fmap = teacher_features.get('pool', teacher_features.get('0'))
        
        if student_fmap is None or teacher_fmap is None:
            return torch.tensor(0.0, device=self.device)

        adapted_student_fmap = self.adaptation_layer(student_fmap)
        loss = torch.nn.functional.mse_loss(adapted_student_fmap, teacher_fmap)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.student(images)
        self.val_map.update(preds, targets)

    def validation_epoch_end(self, outs):
        metrics = self.val_map.compute()
        self.log_dict({"val_AP": metrics["map"], "val_AP50": metrics["map_50"], "val_AP75": metrics["map_75"]}, prog_bar=True)
        self.val_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.student.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 11], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

def main():
    parser = ArgumentParser(description="Knowledge Distillation for Faster R-CNN on MS COCO")
    parser.add_argument("--data_path", type=str, required=True, help="Path to MS COCO dataset root directory.")
    parser.add_argument("--student_weights_path", type=str, default=None, help="Optional: Path to a Lightning checkpoint (.ckpt) for the student model.")
    parser.add_argument("--teacher_weights_path", type=str, default=None, help="Optional: Path to a Lightning checkpoint (.ckpt) for a custom teacher model.")
    parser.add_argument("--student_backbone", type=str, default="resnet34", help="Student backbone ('resnet18' or 'resnet34').")
    parser.add_argument("--teacher_backbone", type=str, default="resnet50", help="Teacher backbone.")
    parser.add_argument("--epochs", type=int, default=12, help="Total number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU.")
    parser.add_argument("--lr", type=float, default=0.02, help="Initial learning rate.")
    parser.add_argument("--wd", type=float, default=0.0001, help="Weight decay.")
    parser.add_argument("--alpha_kd", type=float, default=0.5, help="Weighting factor for the distillation loss.")
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for softening teacher logits.")
    parser.add_argument("--workers", type=int, default=8, help="Number of data loader workers.")
    parser.add_argument("--gpus", type=str, default='0', help="GPU IDs to use (e.g., '0' or '0,1,2').")
    args = parser.parse_args()
    pl.seed_everything(42, workers=True)
    coco_dm = COCODataModule(data_path=args.data_path, batch_size=args.batch_size, workers=args.workers)
    model = KD_FasterRCNN_Lightning(
        student_backbone=args.student_backbone, teacher_backbone=args.teacher_backbone,
        student_weights_path=args.student_weights_path, teacher_weights_path=args.teacher_weights_path,
        lr=args.lr, weight_decay=args.wd, alpha_kd=args.alpha_kd, temperature=args.temperature
    )
    dirpath = f"coco_kd_checkpoints/T_{args.teacher_backbone}-S_{args.student_backbone}/"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_AP", mode="max", dirpath=dirpath, save_top_k=1, filename='best_model-{epoch:02d}-{val_AP:.3f}')
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    if ',' in args.gpus: gpus = [int(i.strip()) for i in args.gpus.split(',')]; strategy = 'ddp'
    else: gpus = int(args.gpus); strategy = None
    trainer = pl.Trainer(gpus=gpus, accelerator="gpu", strategy=strategy, max_epochs=args.epochs,
                         precision=16, callbacks=[checkpoint_callback, lr_monitor], log_every_n_steps=20)
    trainer.fit(model, datamodule=coco_dm)
    print("\nTraining finished."); print(f"Best model checkpoint saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
