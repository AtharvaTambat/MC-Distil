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


class IntervalCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, start_epoch, end_epoch, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def on_validation_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if self.start_epoch <= current_epoch < self.end_epoch:
            super().on_validation_end(trainer, pl_module)


class COCODataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int, workers: int):
        super().__init__()
        self.data_path, self.batch_size, self.workers = data_path, batch_size, workers

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


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root, self.transforms, self.coco = root, transforms, COCO(annotation_file)
        ids = sorted(self.coco.getImgIds())
        self.ids = [i for i in ids if len(self.coco.getAnnIds(imgIds=i, iscrowd=False)) > 0]

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
        boxes, labels = [], []
        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes.append([xmin, ymin, xmin + w, ymin + h])
                labels.append(ann['category_id'])
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)


class FasterRCNN_Lightning(pl.LightningModule):
    def __init__(self, backbone_name: str, lr: float, weight_decay: float, num_classes: int = 91):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._create_model(backbone_name, num_classes)
        self.val_map = MeanAveragePrecision(box_format='xywh')

    def _create_model(self, backbone_name, num_classes):
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(backbone_name, pretrained=True)
        return torchvision.models.detection.FasterRCNN(backbone, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(l for l in loss_dict.values())
        for k, v in loss_dict.items():
            self.log(f'train/{k}', v, on_step=True, prog_bar=False)
        self.log('train/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        self.val_map.update(preds, targets)

    def validation_epoch_end(self, outs):
        metrics = self.val_map.compute()
        self.log_dict({"val_AP": metrics["map"], "val_AP50": metrics["map_50"], "val_AP75": metrics["map_75"]}, prog_bar=True)
        self.val_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def main():
    parser = ArgumentParser()
    parser.add_argument("--backbone", type=str, default="resnet34", help="Backbone ('resnet18' or 'resnet34').")
    parser.add_argument("--data_path", type=str, required=True, help="Path to MS COCO dataset.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--wd", type=float, default=0.0005)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=str, default='0')
    parser.add_argument("--resume", action="store_true", help="Set this flag to resume training from the last checkpoint.")
    args = parser.parse_args()

    pl.seed_everything(42)
    coco_dm = COCODataModule(data_path=args.data_path, batch_size=args.batch_size, workers=args.workers)
    model = FasterRCNN_Lightning(backbone_name=args.backbone, lr=args.lr, weight_decay=args.wd)

    dirpath = f"coco_checkpoints_{args.backbone}/"

    best_ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_AP",
        mode="max",
        dirpath=dirpath,
        filename=f"{args.backbone}-best",
        save_top_k=1,
    )

    last_ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=dirpath,
        filename="last",
        save_top_k=0,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    all_callbacks = [best_ckpt_callback, last_ckpt_callback, lr_monitor]

    resume_ckpt_path = None
    last_ckpt_file = os.path.join(dirpath, "last.ckpt")
    if args.resume and os.path.exists(last_ckpt_file):
        resume_ckpt_path = last_ckpt_file
        print(f"INFO: Resuming training from checkpoint: {resume_ckpt_path}")
    elif args.resume:
        print(f"WARNING: --resume flag was passed, but checkpoint not found at {last_ckpt_file}. Starting from scratch.")

    if ',' in args.gpus:
        gpus = [int(i.strip()) for i in args.gpus.split(',')]
    else:
        gpus = int(args.gpus)
    strategy = 'ddp' if isinstance(gpus, list) and len(gpus) > 1 else None

    trainer = pl.Trainer(
        gpus=gpus,
        accelerator="cuda",
        strategy=strategy,
        max_epochs=args.epochs,
        precision=16,
        callbacks=all_callbacks,
        log_every_n_steps=20,
        resume_from_checkpoint=resume_ckpt_path
    )

    print(f"Starting training for {args.epochs} epochs...")
    trainer.fit(model, coco_dm)
    print("\nTraining finished.")


if __name__ == "__main__":
    main()
