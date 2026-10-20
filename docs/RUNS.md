# sbatch commands

```bash
# FGFM on levels 0-2, 24 ep vs swin_b-moelora (L40S: level 0 likely does not fit on a 3090)
sbatch --gres=gpu:l40s:1 train.slurm --backbone swin_b --init pretrained --adapt moelora --fgfm-levels 0-2 --name swin_b-moelora-fgfm
# Noise floor: 24-ep baseline, seed 1, vs swin_b-moelora (seed 0)
sbatch train.slurm --backbone swin_b --init pretrained --adapt moelora --seed 1 --name swin_b-moelora-s1
# FGFM on levels 2-3 only (strides 16, 32, as the original config's [2,3,4]), 24 ep vs swin_b-moelora and swin_b-moelora-fgfm (levels 0-2)
sbatch train.slurm --backbone swin_b --init pretrained --adapt moelora --fgfm-levels 2-3 --name swin_b-moelora-fgfm23
# NWD anchor labels in the RPN (Wang et al. 2021), c 16, 24 ep vs swin_b-moelora
sbatch train.slurm --backbone swin_b --init pretrained --adapt moelora --rpn-match nwd --name swin_b-moelora-nwd
# max(IoU, NWD) anchor labels in the RPN (NWD for tiny boxes, IoU for large), 24 ep vs swin_b-moelora and swin_b-moelora-nwd
sbatch train.slurm --backbone swin_b --init pretrained --adapt moelora --rpn-match max --name swin_b-moelora-nwdmax
# loss_box_reg x 2, 24 ep vs swin_b-moelora
sbatch train.slurm --backbone swin_b --init pretrained --adapt moelora --loss-weights loss_box_reg=2 --name swin_b-moelora-boxreg2
# Swin-B pretrained at 384 px (12 x 12 windows), 24 ep vs swin_b-moelora
sbatch train.slurm --backbone swin_b384 --init pretrained --adapt moelora --name swin_b384-moelora
# Unfreeze the last Swin-B block (23, stage 4, ~12.6 M; trainable ~31 M), 24 ep vs swin_b-moelora
sbatch train.slurm --backbone swin_b --init pretrained --adapt moelora --unfreeze-blocks 23 --name swin_b-moelora-uf23
# 3 x 3 multi-look averaging as the input instead of raw, 24 ep vs swin_b-moelora
sbatch train.slurm --backbone swin_b --init pretrained --adapt moelora --input-channels ml --name swin_b-moelora-ml
# RetinaNet in place of Faster R-CNN (same anchors and FPN), 24 ep vs swin_b-moelora
sbatch train.slurm --backbone swin_b --init pretrained --adapt moelora --detector retinanet --name swin_b-moelora-retina
# Swin V2-B (in22k -> in1k, 16 x 16 windows) in place of Swin-B, 24 ep vs swin_b-moelora
sbatch --gres=gpu:l40s:1 train.slurm --backbone swinv2_b --init pretrained --adapt moelora --name swinv2_b-moelora
# SimMIM SSL on the train images (no labels): LoRA r64 on every block + block 23 unfrozen, 512 px, 100 ep; if it stops at --max-hours, resubmit with: sbatch train.slurm pretrain_ssl.py --resume runs/<this run>
sbatch train.slurm pretrain_ssl.py --backbone swin_b --rank 64 --unfreeze-blocks 23 --epochs 100 --name swin_b-simmim-r64-uf23
# MoE-LoRA on the SimMIM adapters (frozen, unmerged), 24 ep vs swin_b-moelora
sbatch train.slurm --backbone swin_b --init pretrained --adapt moelora --ssl-weights runs/<swin_b-simmim-r64-uf23 run>/ssl.pt --name swin_b-moelora-simmim
# 768 px input (anchors x 1.5), batch 8 x 2 accumulation (= 16; 512 px peaked at 23.8 GB), 24 ep vs swin_b-moelora; ~2.25x slower, so 10 h
sbatch --gres=gpu:l40s:1 --time=10:00:00 train.slurm --backbone swin_b --init pretrained --adapt moelora --image-size 768 --batch-size 8 --accum-steps 2 --name swin_b-moelora-768
# YOLOv8m (COCO, all 25.9 M trained) + image-class head on SPPF, in place of Swin-B + FPN + Faster R-CNN; AdamW 1e-3, wd 5e-4; 48 ep vs swin_b-moelora-e48-uf01
sbatch train.slurm --backbone yolov8m --init pretrained --adapt full --lr 1e-3 --weight-decay 5e-4 --epochs 48 --name yolov8m-e48
# YOLOv8l (COCO) with layers 0-5 frozen (39.7 M trainable) + image-class head, AdamW 1e-3, wd 5e-4; 48 ep vs yolov8m-e48; ~2x v8m's compute, so 10 h
sbatch --time=10:00:00 train.slurm --backbone yolov8l --init pretrained --adapt full --freeze-layers 6 --lr 1e-3 --weight-decay 5e-4 --epochs 48 --name yolov8l-f6-e48
# YOLO11m (COCO, all ~20 M trained) + image-class head on SPPF; same recipe as yolov8m-e48 (AdamW 1e-3, wd 5e-4, 48 ep), vs yolov8m-e48
sbatch train.slurm --backbone yolo11m --init pretrained --adapt full --lr 1e-3 --weight-decay 5e-4 --epochs 48 --name yolo11m-e48
# YOLO11l (COCO, all ~25 M trained, same size as v8m) + image-class head on SPPF; same recipe as yolov8m-e48 (AdamW 1e-3, wd 5e-4, 48 ep), vs yolov8m-e48
sbatch train.slurm --backbone yolo11l --init pretrained --adapt full --lr 1e-3 --weight-decay 5e-4 --epochs 48 --name yolo11l-e48
# YOLOv8m, same recipe as yolov8m-e48 (AdamW 1e-3, wd 5e-4) at 60 ep (~2 h), vs yolov8m-e48
sbatch train.slurm --backbone yolov8m --init pretrained --adapt full --lr 1e-3 --weight-decay 5e-4 --epochs 60 --name yolov8m-e60
# YOLO11m, same recipe as yolo11m-e48 (AdamW 1e-3, wd 5e-4) at 60 ep, vs yolo11m-e48 and yolov8m-e60
sbatch train.slurm --backbone yolo11m --init pretrained --adapt full --lr 1e-3 --weight-decay 5e-4 --epochs 60 --name yolo11m-e60
# YOLO11s (COCO, all ~9.4 M trained) + image-class head on SPPF; same recipe as yolo11m-e60 (AdamW 1e-3, wd 5e-4, 60 ep), vs yolo11m-e60
sbatch train.slurm --backbone yolo11s --init pretrained --adapt full --lr 1e-3 --weight-decay 5e-4 --epochs 60 --name yolo11s-e60
# YOLOv8m + Quad-FPN in the neck (all 4 parts: deformable convs, CARAFE, space attention, balanced pyramid + non-local; +11.35 M, 37.2 M trainable, each starts as the identity); same recipe as yolov8m-e48, vs yolov8m-e48
sbatch train.slurm --backbone yolov8m --init pretrained --adapt full --quad-fpn all --lr 1e-3 --weight-decay 5e-4 --epochs 48 --name yolov8m-quadfpn-e48
# YOLOv8m + centre keypoint (pose model, 1 keypoint = box centre, Ultralytics' pose loss, gain 12; +0.56 M); same recipe as yolov8m-e48, vs yolov8m-e48; then predict.py with and without --keypoint-centres
sbatch train.slurm --backbone yolov8m --init pretrained --adapt full --centre-keypoint --lr 1e-3 --weight-decay 5e-4 --epochs 48 --name yolov8m-ckpt-e48
# YOLOv8m + random zoom-in crops (half the training images: a square of 50-100 % of the side around one box, resized to 512); same recipe as yolov8m-e48, vs yolov8m-e48
sbatch train.slurm --backbone yolov8m --init pretrained --adapt full --crop-prob 0.5 --crop-min-side 0.5 --lr 1e-3 --weight-decay 5e-4 --epochs 48 --name yolov8m-crop-e48
```
