import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm

from torchmetrics.detection.mean_ap import MeanAveragePrecision


def load_lightning_checkpoint(model, ckpt_path):
    if not os.path.exists(ckpt_path): raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    ckpt_state_dict = checkpoint.get('state_dict', checkpoint)
    cleaned_state_dict = {}
    prefix_found = False
    for prefix in ['student.', 'teacher.', 'proxy.', 'model.']:
        if any(k.startswith(prefix) for k in ckpt_state_dict.keys()):
            for k, v in ckpt_state_dict.items():
                if k.startswith(prefix): cleaned_state_dict[k[len(prefix):]] = v
            prefix_found = True; break
    if not prefix_found: cleaned_state_dict = ckpt_state_dict
    model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"Successfully loaded weights from: {os.path.basename(ckpt_path)}")

class CustomCompose:
    def __init__(self, transforms): 
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms: image, target = t(image, target)
        return image, target
    
class ToTensor:
    def __call__(self, img, target): 
        return torchvision.transforms.functional.to_tensor(img), target

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=None, return_index=False):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annFile)
        self.return_index = return_index
        ids = sorted(self.coco.getImgIds())
        self.ids = [img_id for img_id in ids if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=False)) > 0]

    def __len__(self):
        return len(self.ids)

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
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        if self.return_index:
            return img, target, index
        return img, target

def get_coco_dataloaders(data_path, batch_size):
    train_dir, train_ann = os.path.join(data_path, 'train2017'), os.path.join(data_path, 'annotations/instances_train2017.json')
    val_dir, val_ann = os.path.join(data_path, 'val2017'), os.path.join(data_path, 'annotations/instances_val2017.json')

    train_dataset = COCODataset(root=train_dir, annFile=train_ann, transforms=CustomCompose([ToTensor()]), return_index=True)
    val_dataset = COCODataset(root=val_dir, annFile=val_ann, transforms=CustomCompose([ToTensor()]), return_index=False)
    
    train_loader_stage1 = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: x[0])
    train_loader_stage2 = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda b: tuple(zip(*b)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=lambda b: tuple(zip(*b)))
    
    return train_loader_stage1, train_loader_stage2, val_loader

def create_faster_rcnn_model(backbone_name: str, pretrained_backbone: bool = True, num_classes: int = 91):
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(backbone_name, weights='IMAGENET1K_V1' if pretrained_backbone else None)
    model = torchvision.models.detection.FasterRCNN(backbone, num_classes)
    return model

def create_pruned_models_coco(model_path: str, sparsity_levels: list):
    pruned_models = []
    print("--- Creating Pruned Faster R-CNN Models for Variance Calculation ---")
    for sparsity in sparsity_levels:
        print(f"Creating model with {sparsity*100:.0f}% backbone sparsity...")
        model = create_faster_rcnn_model('resnet50', pretrained_backbone=False)
        load_lightning_checkpoint(model, model_path)
        params_to_prune = [(module, 'weight') for module in model.backbone.modules() if isinstance(module, (nn.Conv2d, nn.Linear))]
        prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=sparsity)
        for module, name in params_to_prune: prune.remove(module, name)
        pruned_models.append(model)
    return pruned_models

def compute_difficulty_coco(args, trainer_kwargs):
    print("--- STAGE 1: Quantifying Sample Difficulty for COCO ---")

    train_loader_stage1, _, _ = get_coco_dataloaders(args.data_path, batch_size=1)

    class Stage1DummyModule(pl.LightningModule):
        def __init__(self, models):
            super().__init__()
            self.models = models

        def forward(self, x):
            return x

    pruned_models = create_pruned_models_coco(args.teacher_ckpt_path, [0.2, 0.4, 0.6, 0.8, 0.9])

    dummy_model = Stage1DummyModule(pruned_models)
    trainer = pl.Trainer(**trainer_kwargs, max_epochs=1)
    device = trainer.strategy.root_device

    num_samples = len(train_loader_stage1.dataset)
    all_losses = torch.zeros(num_samples, len(pruned_models))

    with torch.no_grad():
        for i, model in enumerate(pruned_models):
            print(f"Calculating losses for model {i+1}/{len(pruned_models)}...")
            model.to(device)
            model.train()

            for image, target, index in tqdm(train_loader_stage1, desc=f"Model {i+1}"):
                image_tensor = image.to(device)
                target_tensor = [{k: v.to(device) for k, v in target.items()}]
                loss_dict = model([image_tensor], target_tensor)
                total_loss = sum(loss for loss in loss_dict.values())
                all_losses[index, i] = total_loss.cpu()

    print("\nCalculating variance of losses...")
    loss_variances = torch.var(all_losses, dim=1)
    v_min, v_max = torch.min(loss_variances), torch.max(loss_variances)

    difficulty_degrees = args.alpha + (1 - args.alpha) * (loss_variances - v_min) / (v_max - v_min + 1e-9)

    save_path = "difficulty_scores_coco.pt"
    torch.save(difficulty_degrees, save_path)

    print(f"--- Stage 1 Complete. Difficulty scores for COCO saved to '{save_path}' ---")

class RMC_Distillation_Lightning_COCO(pl.LightningModule):
    def __init__(self, student_model, teacher_model, difficulty_scores, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.student = student_model
        self.teacher = teacher_model
        self.teacher.eval()
        [p.requires_grad_(False) for p in self.teacher.parameters()]
        self.difficulty_scores = difficulty_scores
        self.kd_loss = nn.KLDivLoss(reduction='batchmean')
        self.val_map = MeanAveragePrecision(box_format='xyxy')

        self.beta_reg = 1.0
        self.gamma_hint = 0.5

    def _get_aligned_cls_logits(self, model, features, proposals, image_sizes):
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_sizes)
        return model.roi_heads.box_predictor(model.roi_heads.box_head(box_features))[0]

    def _get_reg_preds(self, model, features, proposals, image_sizes):
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_sizes)
        box_features = model.roi_heads.box_head(box_features)
        _, reg_preds = model.roi_heads.box_predictor(box_features)
        return reg_preds

    def _calculate_hint_loss(self, student_features, teacher_features):
        loss = 0.0
        for (k, s_feat), t_feat in zip(student_features.items(), teacher_features.values()):
            loss += torch.nn.functional.mse_loss(s_feat, t_feat)
        return loss

    def training_step(self, batch, batch_idx):
        images, targets, indices = batch
        loss_dict_student = self.student(images, targets)
        loss_ce = sum(loss for loss in loss_dict_student.values())

        self.teacher.eval()
        with torch.no_grad():
            images_transformed, _ = self.teacher.transform(images, None)
            teacher_features = self.teacher.backbone(images_transformed.tensors)
            teacher_proposals, _ = self.teacher.rpn(images_transformed, teacher_features)
            teacher_logits = self._get_aligned_cls_logits(self.teacher, teacher_features, teacher_proposals, images_transformed.image_sizes)
            teacher_reg_preds = self._get_reg_preds(self.teacher, teacher_features, teacher_proposals, images_transformed.image_sizes)
            teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=1)

            proposals_per_image = [len(p) for p in teacher_proposals]
            d_per_image = self.difficulty_scores[torch.tensor(indices)].to(self.device)
            d_per_proposal = torch.repeat_interleave(d_per_image, torch.tensor(proposals_per_image, device=self.device))
            d = d_per_proposal.unsqueeze(1)

            smoothed_teacher_probs = torch.pow(teacher_probs, d)
            smoothed_teacher_probs /= (smoothed_teacher_probs.sum(dim=1, keepdim=True) + 1e-9)

        student_features = self.student.backbone(images_transformed.tensors)
        student_logits = self._get_aligned_cls_logits(self.student, student_features, teacher_proposals, images_transformed.image_sizes)
        student_log_probs = torch.nn.functional.log_softmax(student_logits / self.hparams.temperature, dim=1)
        student_reg_preds = self._get_reg_preds(self.student, student_features, teacher_proposals, images_transformed.image_sizes)

        loss_kd = self.kd_loss(student_log_probs, smoothed_teacher_probs) * (self.hparams.temperature**2)
        loss_reg_kd = torch.nn.functional.smooth_l1_loss(student_reg_preds, teacher_reg_preds)
        loss_hint = self._calculate_hint_loss(student_features, teacher_features)

        total_kd_loss = loss_kd + (self.beta_reg * loss_reg_kd) + (self.gamma_hint * loss_hint)
        loss = (1 - self.hparams.lambda_kd) * loss_ce + self.hparams.lambda_kd * total_kd_loss

        self.log_dict({
            'train_loss': loss,
            'loss_ce': loss_ce,
            'loss_kd_cls': loss_kd,
            'loss_kd_reg': loss_reg_kd,
            'loss_hint': loss_hint
        }, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.student(images)
        self.val_map.update(preds, targets)

    def on_validation_epoch_end(self):
        metrics = self.val_map.compute()
        self.log_dict({"val_AP": metrics["map"], "val_AP50": metrics["map_50"], "val_AP75": metrics["map_75"]}, prog_bar=True)
        self.val_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 11], gamma=0.1)
        return [optimizer], [scheduler]


def run_distillation_coco(args, trainer_kwargs):
    print("--- STAGE 2: Robust Knowledge Distillation for COCO ---")

    difficulty_scores_path = "difficulty_scores_coco.pt"
    if not os.path.exists(difficulty_scores_path):
        raise FileNotFoundError(f"'{difficulty_scores_path}' not found. Run Stage 1 first.")
    
    difficulty_scores = torch.load(difficulty_scores_path)
    print(f"Loaded difficulty scores for {len(difficulty_scores)} COCO samples.")

    _, train_loader_stage2, val_loader = get_coco_dataloaders(args.data_path, args.batch_size)

    teacher = create_faster_rcnn_model('resnet50', pretrained_backbone=False)
    load_lightning_checkpoint(teacher, args.teacher_ckpt_path)

    student = create_faster_rcnn_model(args.student_model, pretrained_backbone=False)
    load_lightning_checkpoint(student, args.student_ckpt_path)

    model = RMC_Distillation_Lightning_COCO(student, teacher, difficulty_scores, hparams=args)

    dirpath = f"rmc_coco_checkpoints/T_resnet50-S_{args.student_model}/"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_AP',
        mode='max',
        dirpath=dirpath,
        filename='best-student-{epoch}-{val_AP:.3f}'
    )

    trainer = pl.Trainer(
        **trainer_kwargs,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        gradient_clip_val=0.5
    )

    trainer.fit(model, train_dataloaders=train_loader_stage2, val_dataloaders=val_loader)

    final_student_path = os.path.join(dirpath, f'final_student_{args.student_model}_rmc_coco.pth')
    torch.save(student.state_dict(), final_student_path)

    print(f"--- Stage 2 Complete. Final trained student saved to '{final_student_path}' ---")

if __name__ == '__main__':
    parser = ArgumentParser(description="Robust Model Compression for MS COCO Object Detection.")
    
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2], help="Which stage to run.")
    parser.add_argument("--data_path", type=str, required=True, help="Root directory of the MS COCO dataset.")
    parser.add_argument("--gpus", type=str, default='0', help="GPU IDs to use (e.g., '0' or '0,1,2').")
    parser.add_argument("--teacher_ckpt_path", type=str, required=True, help="Path to pre-trained Faster R-CNN ResNet50 checkpoint.")
    parser.add_argument("--student_ckpt_path", type=str, help="Path to pre-trained student model checkpoint (Required for Stage 2).")
    parser.add_argument("--student_model", type=str, default="resnet18", choices=["resnet18", "resnet34"], help="Student backbone architecture.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for Stage 2 training.")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs for Stage 2 training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Stage 2.")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay for Stage 2.")
    parser.add_argument("--alpha", type=float, default=0.3, help="Minimum difficulty degree for smoothing.")
    parser.add_argument("--lambda_kd", type=float, default=0.5, help="Weight for the KD loss term.")
    parser.add_argument("--temperature", type=float, default=3.0, help="Temperature for KD.")
    
    args = parser.parse_args()
    
    try:
        devices = [int(i.strip()) for i in args.gpus.split(',')]
    except ValueError:
        raise ValueError("The '--gpus' argument must be a comma-separated list of integers (e.g., '0' or '0,1').")
    
    strategy = pl.strategies.DDPStrategy(find_unused_parameters=False) if len(devices) > 1 else "auto"
    trainer_kwargs = {
        "devices": devices,
        "accelerator": "gpu",
        "strategy": strategy
    }
    
    if args.stage == 1:
        compute_difficulty_coco(args, trainer_kwargs)
    
    elif args.stage == 2:
        if not args.student_ckpt_path:
            raise ValueError("--student_ckpt_path is required for Stage 2.")
        run_distillation_coco(args, trainer_kwargs)
