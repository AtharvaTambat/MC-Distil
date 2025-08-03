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
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, target):
        if torch.rand(1) < self.p:
            img = torchvision.transforms.functional.hflip(img)
            # Flip the bounding boxes as well
            if 'boxes' in target and target['boxes'].shape[0] > 0:
                img_width = img.shape[2]
                target['boxes'][:, [0, 2]] = img_width - target['boxes'][:, [2, 0]]
        return img, target

class CustomCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class COCODataset(torch.utils.data.Dataset):
    """MS COCO Dataset for Object Detection."""
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation_file)
        ids = sorted(self.coco.getImgIds())
        self.ids = [img_id for img_id in ids if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=False)) > 0]

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        boxes = []
        labels = []
        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            # Ensure the bounding box has a positive area
            if w > 0 and h > 0:
                boxes.append([xmin, ymin, xmin + w, ymin + h])
                labels.append(ann['category_id'])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

class COCODataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for COCO."""
    def __init__(self, data_path: str, batch_size: int, workers: int):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.workers = workers

    def setup(self, stage=None):
        train_dir = os.path.join(self.data_path, 'train2017')
        train_ann = os.path.join(self.data_path, 'annotations/instances_train2017.json')
        val_dir = os.path.join(self.data_path, 'val2017')
        val_ann = os.path.join(self.data_path, 'annotations/instances_val2017.json')
        
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
        if is_train:
            transforms.append(RandomHorizontalFlip())
        return CustomCompose(transforms)

class LearningRateWarmUp(pl.Callback):
    def __init__(self, warmup_steps: int, target_lr: float, initial_lr: float = 1e-6):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.initial_lr = initial_lr

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(trainer.global_step + 1) / self.warmup_steps)
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.initial_lr + lr_scale * (self.target_lr - self.initial_lr)

class DGKD_FasterRCNN_Lightning(pl.LightningModule):
    def __init__(self, student_backbone: str, teacher_backbone: str, ta_backbones: list,
                 student_weights_path: str, teacher_weights_path: str, ta_weights_paths: list,
                 lr: float, weight_decay: float, alpha_kd: float, temperature: float, num_classes: int = 91):
        super().__init__()
        self.save_hyperparameters()

        # Create and load all models
        self.student = self._create_and_load_model(student_backbone, num_classes, student_weights_path, "Student")
        self.teacher = self._create_and_load_model(teacher_backbone, num_classes, teacher_weights_path, "Teacher")
        self._freeze_model(self.teacher)

        self.teacher_assistants = torch.nn.ModuleList()
        for i, ta_bb in enumerate(ta_backbones):
            ta_model = self._create_and_load_model(ta_bb, num_classes, ta_weights_paths[i], f"TA-{i+1} ({ta_bb})")
            self._freeze_model(ta_model)
            self.teacher_assistants.append(ta_model)
        
        self.all_teachers = [self.teacher] + list(self.teacher_assistants)

        self.adaptation_layers = torch.nn.ModuleList()
        student_out_channels = self.student.backbone.out_channels
        
        for i, teacher_model in enumerate(self.all_teachers):
            teacher_name = f"Teacher {i}" if i > 0 else "Main Teacher"
            teacher_out_channels = teacher_model.backbone.out_channels
            print(f"Adapting Student ({student_out_channels} channels) to {teacher_name} ({teacher_out_channels} channels)")

            if student_out_channels == teacher_out_channels:
                layer = torch.nn.Identity()
            else:
                layer = torch.nn.Conv2d(
                    in_channels=student_out_channels, 
                    out_channels=teacher_out_channels, 
                    kernel_size=1
                )
            self.adaptation_layers.append(layer)
        
        self.val_map = MeanAveragePrecision(box_format='xyxy')
        print(f"\nDGKD setup complete. Student: {student_backbone}, Num TAs: {len(self.teacher_assistants)}, Teacher: {teacher_backbone}\n")

    def _create_and_load_model(self, backbone_name, num_classes, ckpt_path, model_name):
        """Helper to create a model and load its weights from a checkpoint."""
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(backbone_name, pretrained=True)
        model = torchvision.models.detection.FasterRCNN(backbone, num_classes)
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"CRITICAL: {model_name} checkpoint file not found: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        ckpt_state_dict = checkpoint.get('state_dict', checkpoint)
        
        model_state_dict = None
        for prefix in ['student.', 'model.']:
            temp_dict = {k.replace(prefix, ''): v for k, v in ckpt_state_dict.items() if k.startswith(prefix)}
            if temp_dict:
                model_state_dict = temp_dict
                break
        
        model.load_state_dict(model_state_dict or ckpt_state_dict, strict=True)
        print(f"Loaded {model_name} ({backbone_name}) weights from {ckpt_path}")
        return model

    @staticmethod
    def _freeze_model(model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, images, targets=None):
        return self.student(images, targets)

    def _get_cls_logits_from_proposals(self, model, features, proposals, image_sizes):
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_sizes)
        box_features = model.roi_heads.box_head(box_features)
        cls_logits, _ = model.roi_heads.box_predictor(box_features)
        return cls_logits

    def _get_reg_preds_from_proposals(self, model, features, proposals, image_sizes):
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_sizes)
        box_features = model.roi_heads.box_head(box_features)
        _, box_regression = model.roi_heads.box_predictor(box_features)
        return box_regression

    def _calculate_hint_loss(self, student_fmap, teacher_fmap, adaptation_layer):
        adapted_student_fmap = adaptation_layer(student_fmap)
        loss = torch.nn.functional.mse_loss(adapted_student_fmap, teacher_fmap)
        return loss

    def training_step(self, batch, batch_idx):
        images, targets = batch
        batch_size = len(images)
        images, targets = self.student.transform(images, targets)
        
        student_features = self.student.backbone(images.tensors)
        proposals, proposal_losses = self.student.rpn(images, student_features, targets)
        _, detector_losses = self.student.roi_heads(student_features, proposals, images.image_sizes, targets)
        loss_ce = sum(loss for loss in {**proposal_losses, **detector_losses}.values())

        with torch.no_grad():
            main_teacher_features = self.teacher.backbone(images.tensors)
            teacher_proposals, _ = self.teacher.rpn(images, main_teacher_features)

            all_teacher_logits, all_teacher_reg_preds, all_teacher_features = [], [], []
            for model in self.all_teachers:
                model_features = model.backbone(images.tensors)
                all_teacher_features.append(model_features)
                all_teacher_logits.append(self._get_cls_logits_from_proposals(model, model_features, teacher_proposals, images.image_sizes))
                all_teacher_reg_preds.append(self._get_reg_preds_from_proposals(model, model_features, teacher_proposals, images.image_sizes))

        T = self.hparams.temperature
        num_teachers = len(self.all_teachers)

        student_cls_logits = self._get_cls_logits_from_proposals(self.student, student_features, teacher_proposals, images.image_sizes)
        total_loss_cls_kd = sum(
            torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_cls_logits / T, dim=1),
                torch.nn.functional.softmax(teacher_logits / T, dim=1), reduction='batchmean'
            ) * (T ** 2) for teacher_logits in all_teacher_logits
        )
        loss_cls_kd_avg = total_loss_cls_kd / num_teachers

        student_reg_preds = self._get_reg_preds_from_proposals(self.student, student_features, teacher_proposals, images.image_sizes)
        total_loss_reg_kd = sum(
            torch.nn.functional.smooth_l1_loss(student_reg_preds, teacher_reg_preds) for teacher_reg_preds in all_teacher_reg_preds
        )
        loss_reg_kd_avg = total_loss_reg_kd / num_teachers
        
        student_fmap = student_features.get('pool', student_features.get('0'))
        total_loss_hint = 0
        if student_fmap is not None:
            for teacher_features, adaptation_layer in zip(all_teacher_features, self.adaptation_layers):
                teacher_fmap = teacher_features.get('pool', teacher_features.get('0'))
                if teacher_fmap is not None:
                    total_loss_hint += self._calculate_hint_loss(student_fmap, teacher_fmap, adaptation_layer)
        loss_hint_avg = total_loss_hint / num_teachers

        beta_reg = 1.0
        gamma_hint = 0.5
        total_distillation_loss = loss_cls_kd_avg + (beta_reg * loss_reg_kd_avg) + (gamma_hint * loss_hint_avg)
        
        loss = (1 - self.hparams.alpha_kd) * loss_ce + self.hparams.alpha_kd * total_distillation_loss
        
        self.log_dict({
            'train/loss': loss, 'train/loss_ce': loss_ce,
            'train/loss_kd_cls': loss_cls_kd_avg, 'train/loss_kd_reg': loss_reg_kd_avg, 'train/loss_hint': loss_hint_avg,
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
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
    parser = ArgumentParser(description="DGKD for Faster R-CNN on MS COCO")
    parser.add_argument("--data_path", type=str, required=True, help="Path to MS COCO dataset root.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume training.")
    parser.add_argument("--student_weights_path", type=str, required=True, help="Path to a pre-trained student .ckpt file.")
    parser.add_argument("--teacher_weights_path", type=str, required=True, help="Path to a pre-trained teacher .ckpt file.")
    parser.add_argument("--ta_weights_paths", type=str, nargs='*', required=True, help="Space-separated paths to pre-trained TA .ckpt files.")
    parser.add_argument("--student_backbone", type=str, default="resnet18", help="Student backbone name.")
    parser.add_argument("--teacher_backbone", type=str, default="resnet50", help="Teacher backbone name.")
    parser.add_argument("--ta_backbones", type=str, nargs='*', required=True, help="Space-separated list of TA backbone names.")
    parser.add_argument("--epochs", type=int, default=12, help="Total training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU.")
    parser.add_argument("--lr", type=float, default=0.001, help="Target learning rate after warm-up.")
    parser.add_argument("--wd", type=float, default=0.0001, help="Weight decay.")
    parser.add_argument("--alpha_kd", type=float, default=0.7, help="Weighting factor for the total KD loss.")
    parser.add_argument("--temperature", type=float, default=5.0, help="Temperature for softening logits.")
    parser.add_argument("--workers", type=int, default=8, help="Number of data loader workers.")
    parser.add_argument("--gpus", type=str, default='0', help="GPU IDs to use (e.g., '0' or '0,1').")
    args = parser.parse_args()

    if len(args.ta_backbones) != len(args.ta_weights_paths):
        raise ValueError("The number of TA backbones must match the number of TA weight paths.")

    pl.seed_everything(42, workers=True)
    coco_dm = COCODataModule(data_path=args.data_path, batch_size=args.batch_size, workers=args.workers)
    
    model = DGKD_FasterRCNN_Lightning(
        student_backbone=args.student_backbone, teacher_backbone=args.teacher_backbone, ta_backbones=args.ta_backbones,
        student_weights_path=args.student_weights_path, teacher_weights_path=args.teacher_weights_path, ta_weights_paths=args.ta_weights_paths,
        lr=args.lr, weight_decay=args.wd, alpha_kd=args.alpha_kd, temperature=args.temperature
    )

    dirpath = f"coco_dgkd_checkpoints/T_{args.teacher_backbone}-S_{args.student_backbone}/"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_AP", mode="max", dirpath=dirpath, save_top_k=3, filename='{epoch:02d}-{val_AP:.3f}')
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    warmup_callback = LearningRateWarmUp(warmup_steps=500, target_lr=args.lr)
    
    if ',' in args.gpus:
        gpus = [int(i.strip()) for i in args.gpus.split(',')]
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=False)
    else:
        gpus = int(args.gpus)
        strategy = None

    trainer = pl.Trainer(gpus=gpus, accelerator="gpu", strategy=strategy, max_epochs=args.epochs,
                         precision=16, callbacks=[checkpoint_callback, lr_monitor, warmup_callback], log_every_n_steps=20)
    
    trainer.fit(model, datamodule=coco_dm, ckpt_path=args.resume_from)
    
    print(f"\nTraining finished. Best model checkpoint saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()