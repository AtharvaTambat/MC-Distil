import os
import torch
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, Subset
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
    
class CustomCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class COCODataModule(pl.LightningDataModule):
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

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, ann_file, transforms=None):
        self.root = root; self.transforms = transforms; self.coco = COCO(ann_file)
        ids = sorted(self.coco.getImgIds())
        self.ids = [i for i in ids if self.coco.getAnnIds(imgIds=i, iscrowd=False)]

    def __len__(self): return len(self.ids)

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
                 student18_ckpt: str, student34_ckpt: str, teacher_ckpt: str,
                 num_classes: int = 91):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.student_names = ['resnet18','resnet34']
        self.students = torch.nn.ModuleList([self._make_frcnn(n) for n in self.student_names])
        for idx, path in enumerate([student18_ckpt, student34_ckpt]):
            assert os.path.exists(path), f"Student checkpoint not found: {path}"
            self._load_ckpt(self.students[idx], path)
            
        self.teacher = self._make_frcnn('resnet50')
        assert os.path.exists(teacher_ckpt), f"Teacher checkpoint not found: {teacher_ckpt}"
        self._load_ckpt(self.teacher, teacher_ckpt)
        self.teacher.eval()

        self.adaptation_layers = torch.nn.ModuleList()
        teacher_out_channels = self.teacher.backbone.out_channels
        for student in self.students:
            student_out_channels = student.backbone.out_channels
            if student_out_channels == teacher_out_channels:
                # If channels match, use an identity mapping
                self.adaptation_layers.append(torch.nn.Identity())
            else:
                # Otherwise, use a 1x1 convolution to adapt channels
                self.adaptation_layers.append(
                    torch.nn.Conv2d(student_out_channels, teacher_out_channels, kernel_size=1)
                )

        meta_backbone = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
        self.meta_net = torch.nn.Sequential(*list(meta_backbone.children())[:-1])
        self.meta_head = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(512, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, len(self.students) * 2), torch.nn.Softplus()
        )
        self.val_maps = torch.nn.ModuleList([MeanAveragePrecision(box_format='xyxy') for _ in self.students])

    def _make_frcnn(self, backbone_name: str) -> FasterRCNN:
        bb = resnet_fpn_backbone(backbone_name, pretrained=True)
        return FasterRCNN(bb, self.hparams.num_classes)

    def _load_ckpt(self, model: torch.nn.Module, path: str):
        ck = torch.load(path, map_location='cpu')['state_dict']
        new_sd = {k.replace('model.', ''): v for k, v in ck.items()}
        model.load_state_dict(new_sd)

    def forward(self, imgs, idx=0): return self.students[idx](imgs)

    def training_step(self, batch, batch_idx):
        student_opts = self.optimizers()[:-1]
        meta_opt = self.optimizers()[-1]

        imgs, tgts = batch
        imgs = [i.to(self.device) for i in imgs]
        tgts = [{k: v.to(self.device) for k, v in t.items()} for t in tgts]
        imgs_t, _ = self.students[0].transform(imgs, tgts)

        if self.global_step % self.hparams.meta_interval == 0:
            feat = self.meta_net(imgs_t.tensors.to(self.device))
            all_w = self.meta_head(feat)

            loss_meta = 0
            for i, stu in enumerate(self.students):
                w = all_w[:, i*2:(i+1)*2]
                l, _ = self._calculate_weighted_loss(stu, self.adaptation_layers[i], imgs_t, tgts, self.teacher, w)
                loss_meta += l
            
            meta_opt.zero_grad()
            self.manual_backward(loss_meta)
            self.clip_gradients(meta_opt, gradient_clip_val=self.hparams.grad_clip, gradient_clip_algorithm="norm")
            meta_opt.step()
            self.log('train/meta_loss', loss_meta, on_step=True)

        with torch.no_grad():
            feat_detached = self.meta_net(imgs_t.tensors)
            all_w_detached = self.meta_head(feat_detached)

        for i, (stu, opt) in enumerate(zip(self.students, student_opts)):
            opt.zero_grad()
            w = all_w_detached[:, i*2:(i+1)*2]
            loss, logs = self._calculate_weighted_loss(stu, self.adaptation_layers[i], imgs_t, tgts, self.teacher, w)
            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=self.hparams.grad_clip, gradient_clip_algorithm="norm")
            opt.step()
            self.log(f'train/S{i}_total_loss', loss, on_step=True)
            self.log(f'train/S{i}_hard_loss', logs['hard_loss'], on_step=True)
            self.log(f'train/S{i}_soft_loss', logs['soft_loss'], on_step=True)

    def _calculate_feature_loss(self, student_feats, teacher_feats, adaptation_layer):
        student_fmap = student_feats.get('pool')
        teacher_fmap = teacher_feats.get('pool')

        if student_fmap is None or teacher_fmap is None:
            return torch.tensor(0.0, device=self.device)

        adapted_student_fmap = adaptation_layer(student_fmap)
        
        return torch.nn.functional.mse_loss(adapted_student_fmap, teacher_fmap)
    
    def _get_cls_logits(self, model, features, proposals, image_sizes):
        # Pass features and proposals through the ROI head to get classification logits
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_sizes)
        box_features = model.roi_heads.box_head(box_features)
        cls_logits, _ = model.roi_heads.box_predictor(box_features)
        return cls_logits

    def _get_reg_preds(self, model, features, proposals, image_sizes):
        # Pass features and proposals through the ROI head to get regression predictions
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_sizes)
        box_features = model.roi_heads.box_head(box_features)
        _, box_regression = model.roi_heads.box_predictor(box_features)
        return box_regression

    def _calculate_weighted_loss(self, student, adaptation_layer, images, targets, teacher, weights):
        feats_s = student.backbone(images.tensors)
        props_s, loss_rpn_dict = student.rpn(images, feats_s, targets)
        _, loss_roi_dict = student.roi_heads(feats_s, props_s, images.image_sizes, targets)
        
        # Hard Loss
        loss_hard = sum(loss for loss in loss_rpn_dict.values()) + sum(loss for loss in loss_roi_dict.values())

        # Teacher Forward Pass
        with torch.no_grad():
            feats_t = teacher.backbone(images.tensors)
            props_t, _ = teacher.rpn(images, feats_t)
            logits_t = self._get_cls_logits(teacher, feats_t, props_t, images.image_sizes)
            reg_preds_t = self._get_reg_preds(teacher, feats_t, props_t, images.image_sizes)

        logits_s = self._get_cls_logits(student, feats_s, props_t, images.image_sizes)
        reg_preds_s = self._get_reg_preds(student, feats_s, props_t, images.image_sizes)
        
        T = self.hparams.temperature
        # Classification KD Loss
        loss_kd_cls = T**2 * torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits_s / T, 1),
            torch.nn.functional.softmax(logits_t / T, 1),
            reduction='batchmean'
        )
        
        # Regression KD Loss
        loss_kd_reg = torch.nn.functional.smooth_l1_loss(reg_preds_s, reg_preds_t)

        # Feature Distillation Loss (Hint Loss)
        loss_kd_feat = self._calculate_feature_loss(feats_s, feats_t, adaptation_layer)

        loss_soft = loss_kd_cls + loss_kd_reg + loss_kd_feat
        
        w_hard, w_soft = weights[:,0], weights[:,1]
        total_loss = w_hard.mean() * loss_hard + w_soft.mean() * loss_soft
        
        # Return loss and a dictionary for detailed logging
        logs = {'hard_loss': loss_hard, 'soft_loss': loss_soft}
        return total_loss, logs
    
    def validation_step(self, batch, batch_idx):
        imgs, tgts = batch
        imgs = [img.to(self.device) for img in imgs]
        tgts = [{k: v.to(self.device) for k, v in t.items()} for t in tgts]
        for i, stu in enumerate(self.students):
            stu.eval()
            with torch.no_grad():
                preds = stu(imgs)
            self.val_maps[i].update(preds, tgts)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            print("\n" + "#"*30 + f" End of Epoch {self.current_epoch} " + "#"*30)
            for i, name in enumerate(self.student_names):
                metrics = self.val_maps[i].compute()
                ap   = metrics["map"]
                ap50 = metrics["map_50"]
                ap75 = metrics["map_75"]
                self.log(f"val/map_{name}",   ap, on_epoch=True, prog_bar=True)
                self.log(f"val/map50_{name}", ap50, on_epoch=True)
                self.log(f"val/map75_{name}", ap75, on_epoch=True)
                print(f"[Epoch {self.current_epoch:02d}] {name} → "
                      f"AP:   {ap:.4f}, "
                      f"AP50: {ap50:.4f}, "
                      f"AP75: {ap75:.4f}")
                self.val_maps[i].reset()
            print("#"*80 + "\n")

    def configure_optimizers(self):
        studs = [torch.optim.Adam(s.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay) for s in self.students]
        meta  = torch.optim.Adam(self.meta_net.parameters(), lr=self.hparams.meta_lr)
        return studs + [meta]

def main():
    pl.seed_everything(42)
    parser = ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--meta_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--meta_interval', type=int, default=10, help="Update meta-network every N steps.")
    parser.add_argument('--student18_ckpt', type=str, required=True)
    parser.add_argument('--student34_ckpt', type=str, required=True)
    parser.add_argument('--teacher_ckpt', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()

    dm = COCODataModule(args.data_path, args.batch_size, args.workers)
    model = MetaKD_FRCNN(
        lr=args.lr, meta_lr=args.meta_lr, weight_decay=args.weight_decay,
        temperature=args.temperature, grad_clip=args.grad_clip,
        meta_interval=args.meta_interval,
        student18_ckpt=args.student18_ckpt, student34_ckpt=args.student34_ckpt,
        teacher_ckpt=args.teacher_ckpt
    )
    ckpt_cb = pl.callbacks.ModelCheckpoint(monitor='val/map_resnet34', mode='max', save_top_k=2)
    
    devices = [int(x.strip()) for x in args.gpus.split(',')] if ',' in args.gpus else [int(args.gpus)]
    
    if len(devices) > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = None

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
