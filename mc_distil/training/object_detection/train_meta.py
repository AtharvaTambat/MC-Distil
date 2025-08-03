import os
import torch
import copy
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser
from pycocotools.coco import COCO
from PIL import Image

from pytorch_lightning.strategies import DDPStrategy
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class ToTensor:
    def __call__(self, img, target):
        return torchvision.transforms.functional.to_tensor(img), target

class RandomHorizontalFlip:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img, target):
        if torch.rand(1) < self.p:
            img = torchvision.transforms.functional.hflip(img)
            if 'boxes' in target and target['boxes'].shape[0] > 0:
                w = img.shape[2]
                target['boxes'][:, [0,2]] = w - target['boxes'][:, [2,0]]
        return img, target

class ComposeTM:
    def __init__(self, ts): self.ts = ts
    def __call__(self, img, t):
        for tr in self.ts: img, t = tr(img, t)
        return img, t

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, ann_file, transforms=None):
        self.root = root; self.transforms = transforms; self.coco = COCO(ann_file)
        ids = sorted(self.coco.getImgIds())
        self.ids = [i for i in ids if self.coco.getAnnIds(imgIds=i, iscrowd=False)]

    def __len__(self): 
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.root, info['file_name'])
        img = Image.open(path).convert('RGB')
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels = [], []
        for ann in anns:
            x,y,w,h = ann['bbox']
            if w>0 and h>0:
                boxes.append([x, y, x+w, y+h]); labels.append(ann['category_id'])
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4)),
            'labels': torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([img_id])
        }
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

class MetaKD_FRCNN(pl.LightningModule):
    def __init__(self,
                 lr: float, meta_lr: float, weight_decay: float,
                 temperature: float, grad_clip: float, meta_interval: int,
                 student_backbone: str, student_ckpt: str, teacher_ckpt: str,
                 num_classes: int = 91):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.meta_loader_iter = None

        self.student_name = student_backbone
        self.student = self._make_frcnn(self.student_name)
        assert os.path.exists(student_ckpt), f"Student checkpoint not found: {student_ckpt}"
        self._load_ckpt(self.student, student_ckpt)
            
        self.teacher = self._make_frcnn('resnet50')
        assert os.path.exists(teacher_ckpt), f"Teacher checkpoint not found: {teacher_ckpt}"
        self._load_ckpt(self.teacher, teacher_ckpt)
        self.teacher.eval()

        student_out_channels = self.student.backbone.out_channels
        teacher_out_channels = self.teacher.backbone.out_channels
        if student_out_channels != teacher_out_channels:
            self.hint_adaptor = torch.nn.Conv2d(student_out_channels, teacher_out_channels, kernel_size=1)
        else:
            self.hint_adaptor = torch.nn.Identity()

        meta_backbone = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
        self.meta_net = torch.nn.Sequential(*list(meta_backbone.children())[:-1])
        self.meta_head = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(512, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 2), torch.nn.Softplus()
        )
        self.val_map = MeanAveragePrecision(box_format='xyxy')

    def _make_frcnn(self, backbone_name: str) -> FasterRCNN:
        bb = resnet_fpn_backbone(backbone_name, weights=None)
        return FasterRCNN(bb, self.hparams.num_classes)

    def _load_ckpt(self, model: torch.nn.Module, path: str):
        ck = torch.load(path, map_location='cpu')['state_dict']
        new_sd = {k.replace('model.', ''): v for k, v in ck.items()}
        model.load_state_dict(new_sd)

    def forward(self, imgs): 
        return self.student(imgs)

    def on_train_epoch_start(self):
        if self.trainer.datamodule:
            self.meta_loader_iter = iter(self.trainer.datamodule.meta_val_dataloader())

    def training_step(self, batch, batch_idx):
        student_opt, meta_opt = self.optimizers()
        
        imgs, tgts = batch
        imgs = [i.to(self.device) for i in imgs]
        tgts = [{k: v.to(self.device) for k, v in t.items()} for t in tgts]
        imgs_t, _ = self.student.transform(imgs, tgts)

        if self.global_step > 0 and self.global_step % self.hparams.meta_interval == 0:
            pseudo_net = copy.deepcopy(self.student)
            pseudo_net.train()
            
            feat = self.meta_net(imgs_t.tensors)
            weights = self.meta_head(feat)
            
            pseudo_loss, _ = self._calculate_weighted_loss(pseudo_net, imgs_t, tgts, self.teacher, weights)

            trainable_pseudo_params = [p for p in pseudo_net.parameters() if p.requires_grad]
            pseudo_grads = torch.autograd.grad(pseudo_loss, trainable_pseudo_params, create_graph=True, allow_unused=True)
            
            student_lr = student_opt.param_groups[0]['lr']
            for p, g in zip(trainable_pseudo_params, pseudo_grads):
                if g is not None:
                    p.data.sub_(g, alpha=student_lr)

            try:
                meta_imgs, meta_tgts = next(self.meta_loader_iter)
            except StopIteration:
                self.meta_loader_iter = iter(self.trainer.datamodule.meta_val_dataloader())
                meta_imgs, meta_tgts = next(self.meta_loader_iter)

            meta_imgs = [i.to(self.device) for i in meta_imgs]
            meta_tgts = [{k: v.to(self.device) for k, v in t.items()} for t in meta_tgts]
            meta_imgs_t, _ = pseudo_net.transform(meta_imgs, meta_tgts)
            
            meta_loss = self._calculate_ce_loss(pseudo_net, meta_imgs_t, meta_tgts)

            meta_opt.zero_grad()
            self.manual_backward(meta_loss)
            self.clip_gradients(meta_opt, gradient_clip_val=self.hparams.grad_clip, gradient_clip_algorithm="norm")
            meta_opt.step()
            self.log('train/meta_loss', meta_loss, on_step=True)

        with torch.no_grad():
            feat_detached = self.meta_net(imgs_t.tensors)
            weights_detached = self.meta_head(feat_detached)
        
        student_opt.zero_grad()
        student_loss, loss_components = self._calculate_weighted_loss(self.student, imgs_t, tgts, self.teacher, weights_detached)
        self.manual_backward(student_loss)
        self.clip_gradients(student_opt, gradient_clip_val=self.hparams.grad_clip, gradient_clip_algorithm="norm")
        student_opt.step()
        
        self.log_dict({
            'train/total_loss': student_loss,
            'train/loss_hard': loss_components['hard'],
            'train/loss_soft': loss_components['soft'],
            'train/loss_cls_kd': loss_components['cls_kd'],
            'train/loss_reg_kd': loss_components['reg_kd'],
            'train/loss_hint': loss_components['hint']
        }, on_step=True, on_epoch=True, prog_bar=True)

    def _get_cls_logits(self, model, features, proposals, image_shapes):
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
        box_features = model.roi_heads.box_head(box_features)
        cls_logits, _ = model.roi_heads.box_predictor(box_features)
        return cls_logits

    def _get_reg_preds(self, model, features, proposals, image_shapes):
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
        box_features = model.roi_heads.box_head(box_features)
        _, reg_preds = model.roi_heads.box_predictor(box_features)
        return reg_preds

    def _calculate_hint_loss(self, student_features, teacher_features):
        student_hint = student_features.get('pool', None)
        teacher_hint = teacher_features.get('pool', None)
        
        if student_hint is None or teacher_hint is None:
            return 0.0
            
        adapted_student_hint = self.hint_adaptor(student_hint)
        return torch.nn.functional.mse_loss(adapted_student_hint, teacher_hint)

    def _calculate_ce_loss(self, student, images, targets):
        student.train()
        features = student.backbone(images.tensors)
        proposals, _ = student.rpn(images, features, targets)
        _, roi_loss_dict = student.roi_heads(features, proposals, images.image_sizes, targets)
        ce_loss = roi_loss_dict['loss_classifier'] + roi_loss_dict['loss_box_reg']
        return ce_loss

    def _calculate_weighted_loss(self, student, images, targets, teacher, weights):
        student.train()
        student_features = student.backbone(images.tensors)
        
        # Hard Loss
        proposals, p_loss = student.rpn(images, student_features, targets)
        _, d_loss = student.roi_heads(student_features, proposals, images.image_sizes, targets)
        loss_hard = sum(loss for loss in {**p_loss, **d_loss}.values())

        # Soft Loss
        with torch.no_grad():
            teacher_features = teacher.backbone(images.tensors)
            teacher_proposals, _ = teacher.rpn(images, teacher_features)
            t_props_matched, _, _, _ = teacher.roi_heads.select_training_samples(teacher_proposals, targets)
            teacher_cls_logits = self._get_cls_logits(teacher, teacher_features, t_props_matched, images.image_sizes)
            teacher_reg_preds = self._get_reg_preds(teacher, teacher_features, t_props_matched, images.image_sizes)

        # Classification Distillation
        student_cls_logits = self._get_cls_logits(student, student_features, t_props_matched, images.image_sizes)
        T = self.hparams.temperature
        loss_cls_kd = T**2 * torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_cls_logits / T, dim=1),
            torch.nn.functional.softmax(teacher_cls_logits / T, dim=1),
            reduction='batchmean'
        )

        # Regression Distillation
        student_reg_preds = self._get_reg_preds(student, student_features, t_props_matched, images.image_sizes)
        loss_reg_kd = torch.nn.functional.smooth_l1_loss(student_reg_preds, teacher_reg_preds)

        # Feature Hint Distillation
        loss_hint = self._calculate_hint_loss(student_features, teacher_features)

        loss_soft = loss_cls_kd + loss_reg_kd + loss_hint
        
        w_hard, w_soft = weights[:, 0], weights[:, 1]
        total_loss = w_hard.mean() * loss_hard + w_soft.mean() * loss_soft
        
        loss_components = {
            'hard': loss_hard, 'soft': loss_soft,
            'cls_kd': loss_cls_kd, 'reg_kd': loss_reg_kd, 'hint': loss_hint
        }
        return total_loss, loss_components

    def validation_step(self, batch, batch_idx):
        imgs, tgts = batch
        self.student.eval()
        with torch.no_grad():
            preds = self.student(imgs)
        self.val_map.update(preds, tgts)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            print("\n" + "#"*30 + f" End of Epoch {self.current_epoch} " + "#"*30)
            metrics = self.val_map.compute()
            ap, ap50, ap75 = metrics["map"], metrics["map_50"], metrics["map_75"]
            self.log(f"val/map_{self.student_name}", ap, on_epoch=True, prog_bar=True)
            self.log(f"val/map50_{self.student_name}", ap50, on_epoch=True)
            self.log(f"val/map75_{self.student_name}", ap75, on_epoch=True)
            print(f"[Epoch {self.current_epoch:02d}] {self.student_name} → AP: {ap:.4f}, AP50: {ap50:.4f}, AP75: {ap75:.4f}")
            self.val_map.reset()
            print("#"*80 + "\n")

    def configure_optimizers(self):
        student_params = list(self.student.parameters())
        student_params.extend(list(self.hint_adaptor.parameters()))
        student_opt = torch.optim.Adam(student_params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        meta_params = list(self.meta_net.parameters()) + list(self.meta_head.parameters())
        meta_opt  = torch.optim.Adam(meta_params, lr=self.hparams.meta_lr)
        
        return student_opt, meta_opt

class COCODataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, workers, seed=42):
        super().__init__(); self.save_hyperparameters()

    def setup(self, stage=None):
        d = self.hparams.data_path
        if stage == 'fit' or stage is None:
            full_train = COCODataset(os.path.join(d, 'train2017'), os.path.join(d, 'annotations/instances_train2017.json'),
                                     transforms=ComposeTM([ToTensor(), RandomHorizontalFlip()]))
            
            g = torch.Generator().manual_seed(self.hparams.seed)
            n_meta = len(full_train) // 10
            n_train = len(full_train) - n_meta
            self.train_ds, self.meta_ds = random_split(full_train, [n_train, n_meta], generator=g)

        self.val_ds = COCODataset(os.path.join(d, 'val2017'), os.path.join(d, 'annotations/instances_val2017.json'),
                                  transforms=ComposeTM([ToTensor()]))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.workers, collate_fn=lambda b: tuple(zip(*b)))

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.workers, collate_fn=lambda b: tuple(zip(*b)))

    def meta_val_dataloader(self):
        return DataLoader(self.meta_ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.workers, collate_fn=lambda b: tuple(zip(*b)))
    
def main():
    pl.seed_everything(42)
    parser = ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--gpus', type=str, default='0')
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--meta_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--meta_interval', type=int, default=10)
    
    parser.add_argument('--student_backbone', type=str, required=True, choices=['resnet18', 'resnet34'])
    parser.add_argument('--student_ckpt', type=str, required=True)
    parser.add_argument('--teacher_ckpt', type=str, required=True)
    
    args = parser.parse_args()

    dm = COCODataModule(args.data_path, args.batch_size, args.workers)
    
    model = MetaKD_FRCNN(
        lr=args.lr, meta_lr=args.meta_lr, weight_decay=args.weight_decay,
        temperature=args.temperature, grad_clip=args.grad_clip,
        meta_interval=args.meta_interval,
        student_backbone=args.student_backbone,
        student_ckpt=args.student_ckpt,
        teacher_ckpt=args.teacher_ckpt
    )
    
    monitor_metric = f'val/map_{args.student_backbone}'
    ckpt_cb = pl.callbacks.ModelCheckpoint(monitor=monitor_metric, mode='max', save_top_k=2)
    
    devices = [int(x.strip()) for x in args.gpus.split(',')] if ',' in args.gpus else [int(args.gpus)]
    strategy = DDPStrategy(find_unused_parameters=True) if len(devices) > 1 else None

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=devices,
        strategy=strategy,
        max_epochs=args.epochs,
        callbacks=[ckpt_cb],
        log_every_n_steps=50
    )

    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    main()