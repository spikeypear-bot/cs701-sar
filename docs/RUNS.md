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
# YOLOv8m at 800 px input (--image-size 800, 2.4x the pixels of 512); same recipe as yolov8m-e48 (AdamW 1e-3, wd 5e-4, 48 ep), vs yolov8m-e48; ~2.4x compute, so 10 h
sbatch --time=10:00:00 train.slurm --backbone yolov8m --init pretrained --adapt full --image-size 800 --lr 1e-3 --weight-decay 5e-4 --epochs 48 --name yolov8m-800-e48
# YOLOv8m-P2 (yolov8-p2.yaml: 4th Detect level at stride 4; COCO weights in layers 0-15 only, the P2 neck and head new; 25.1 M), 512 px; same recipe as yolov8m-e48, vs yolov8m-e48; ~1.25x compute
sbatch train.slurm --backbone yolov8m-p2 --init pretrained --adapt full --lr 1e-3 --weight-decay 5e-4 --epochs 48 --name yolov8m-p2-e48
# Ultralytics' own trainer + default recipe (EMA, mosaic, scale/translate, warmup, AdamW auto lr), YOLOv8m COCO, 512 px, 48 ep, detection only; holdout scored as train.py does; vs yolov8m-e48 (41.89 mAP)
sbatch --mem=64GB train.slurm ultralytics_baseline.py --model yolov8m --epochs 48 --name ul-yolov8m-e48
# ConvNeXt-B (ImageNet-22k) + FPN + Faster R-CNN, the ConvNeXt paper's detection recipe (AdamW 1e-4, wd 0.05, drop path 0.6, 36 ep, lr x0.1 at 27 and 33); MoE-LoRA on every block's MLP + last 2 blocks (stage 4) trained at 8e-5 (= 1e-4 x the paper's layer decay 0.8), ~37.8 M trainable; vs swin_b-moelora-e36 (33.2)
sbatch --gres=gpu:l40s:1 --time=08:00:00 train.slurm --backbone convnext_b --init pretrained --adapt moelora --unfreeze-blocks 34-35 --unfreeze-lr 8e-5 --lr 1e-4 --weight-decay 0.05 --drop-path 0.6 --schedule step --epochs 36 --name convnext_b-moelora-uf34-e36
# YOLOv8m all-on: weight EMA 0.9999 + YOLO26 small-box assignment (16 px) + image-class head on the neck maps (MLP, dropout 0.2) + label smoothing 0.1; yolov8m-e48 recipe otherwise, vs yolov8m-e48 (41.89 mAP, 93.8 F1); then leave-one-out
sbatch train.slurm --backbone yolov8m --init pretrained --adapt full --lr 1e-3 --weight-decay 5e-4 --epochs 48 --ema 0.9999 --small-box-assign 16 --cls-pool neck --cls-head mlp --cls-dropout 0.2 --label-smoothing 0.1 --name yolov8m-allon-e48
# YOLOv8m with SGD (Nesterov, momentum 0.937, lr 0.01, no weight decay on 1-D parameters, as Ultralytics), yolov8m-e48 recipe otherwise, vs yolov8m-e48 (AdamW 1e-3)
sbatch train.slurm --backbone yolov8m --init pretrained --adapt full --optimizer sgd --lr 1e-2 --weight-decay 5e-4 --epochs 48 --name yolov8m-sgd-e48
# YOLOv8m 200 epochs, yolov8m-e100 recipe otherwise (e48 41.89, e60 42.85, e100 44.80, Codabench +32); 8 h limit for headroom
sbatch --time=08:00:00 train.slurm --backbone yolov8m --init pretrained --adapt full --lr 1e-3 --weight-decay 5e-4 --epochs 200 --name yolov8m-e200
# YOLOv8m box loss gain x1.5 (7.5 -> 11.25), for tighter boxes (AP/AP50 0.64); vs yolov8m-e48
sbatch train.slurm --backbone yolov8m --init pretrained --adapt full --lr 1e-3 --weight-decay 5e-4 --epochs 48 --loss-weights loss_yolo_box=1.5 --name yolov8m-box1.5-e48
# YOLOv8m DFL gain x2 (1.5 -> 3); vs yolov8m-e48
sbatch train.slurm --backbone yolov8m --init pretrained --adapt full --lr 1e-3 --weight-decay 5e-4 --epochs 48 --loss-weights loss_yolo_dfl=2 --name yolov8m-dfl2-e48
# YOLOv8m 200 epochs + rot90 + same-class BoxPaste (P 0.5, 1-3 objects <= 32 px of ship/turbine/oil tank/car/aircraft); size-preserving augmentation first tested on YOLO and on a long run; vs yolov8m-e200
sbatch --time=08:00:00 train.slurm --backbone yolov8m --init pretrained --adapt full --lr 1e-3 --weight-decay 5e-4 --epochs 200 --rot90 --boxpaste-prob 0.5 --name yolov8m-aug-e200
# yolov8m-aug-e200 + brightness/contrast jitter (each uniform in 1 +- 0.2); vs yolov8m-aug-e200
sbatch --time=08:00:00 train.slurm --backbone yolov8m --init pretrained --adapt full --lr 1e-3 --weight-decay 5e-4 --epochs 200 --rot90 --boxpaste-prob 0.5 --brightness 0.2 --name yolov8m-aug-bright-e200
# YOLOv8m e100, AdamW weight decay 0.01 on weights only (--no-decay-1d: no decay on norm weights and biases); yolov8m-e100 recipe otherwise (wd 5e-4 on every parameter, 44.80 mAP), vs yolov8m-e100
sbatch train.slurm --backbone yolov8m --init pretrained --adapt full --lr 1e-3 --weight-decay 0.01 --no-decay-1d --epochs 100 --name yolov8m-wd0.01-e100
# YOLOv8m e100, AdamW weight decay 0.05 on weights only (--no-decay-1d); vs yolov8m-e100 and yolov8m-wd0.01-e100
sbatch train.slurm --backbone yolov8m --init pretrained --adapt full --lr 1e-3 --weight-decay 0.05 --no-decay-1d --epochs 100 --name yolov8m-wd0.05-e100
# YOLO26x-P2 (Ultralytics 8.4.189 trainer and default recipe, env cs701y26), COCO yolo26x weights where shapes fit, layers 5-8 frozen (40 M limit), Albumentations multiplicative noise 1 +- 0.1, gamma 80-120, brightness/contrast 0.2 (p 0.5 each), 100 ep; image class = top box's class; as the groupmate's run (Codabench dm +41.16), vs yolov8m-e100
sbatch --mem=64GB --time=1-00:00:00 train.slurm yolo26.py --freeze 5-8 --epochs 100 --mult-noise 0.1 --gamma 0.2 --brightness-contrast 0.2 --name y26x-p2-f5-8-e100
```
# y26x-p2 e100 screen G: Ultralytics scale 0.5 -> 0.25 (less shrinking of tiny objects; oil tank -7.9), vs y26x-p2-f5-8-e100
sbatch --mem=64GB --time=1-00:00:00 train.slurm yolo26.py --freeze 5-8 --epochs 100 --mult-noise 0.1 --gamma 0.2 --brightness-contrast 0.2 --hyp scale=0.25 --name y26x-p2-f5-8-scale025-e100
# y26x-p2 e100 screen L: box loss weight 7.5 -> 12 (loose boxes: AP@0.5 70.5 vs AP 45.5), vs y26x-p2-f5-8-e100
sbatch --mem=64GB --time=1-00:00:00 train.slurm yolo26.py --freeze 5-8 --epochs 100 --mult-noise 0.1 --gamma 0.2 --brightness-contrast 0.2 --hyp box=12 --name y26x-p2-f5-8-box12-e100
# y26x-p2 e100 screen F: vertical flips on (Ultralytics flipud 0 -> 0.5; our own trainer always used both flips), vs y26x-p2-f5-8-e100
sbatch --mem=64GB --time=1-00:00:00 train.slurm yolo26.py --freeze 5-8 --epochs 100 --mult-noise 0.1 --gamma 0.2 --brightness-contrast 0.2 --hyp flipud=0.5 --name y26x-p2-f5-8-flipud-e100
# y26x-p2 e100 screen P: ProgLoss one-to-many weight kept at 0.8 (Ultralytics decays it to 0.1; only loss reaching backbone + neck), vs y26x-p2-f5-8-e100
sbatch --mem=64GB --time=1-00:00:00 train.slurm yolo26.py --freeze 5-8 --epochs 100 --mult-noise 0.1 --gamma 0.2 --brightness-contrast 0.2 --o2m-final 0.8 --name y26x-p2-f5-8-o2m08-e100
# y26x-p2 e100 screen I: bridge/playground train boxes with no bright return (943 + 390, finding 21) give no loss (not targets, not background), vs y26x-p2-f5-8-e100
sbatch --mem=64GB --time=1-00:00:00 train.slurm yolo26.py --freeze 5-8 --epochs 100 --mult-noise 0.1 --gamma 0.2 --brightness-contrast 0.2 --ignore-dim bridge,playground --name y26x-p2-f5-8-ignoredim-e100
# y26x-p2 e100 augmentation bundle (seed 0): per-class scale + per-pixel speckle + JPEG 50-100, vs y26x-p2-f5-8-scale025-e100
sbatch --mem=64GB --time=1-00:00:00 train.slurm yolo26.py --freeze 5-8 --epochs 100 --mult-noise 0.1 --gamma 0.2 --brightness-contrast 0.2 --elementwise --jpeg 50 --class-scale default=0.25,ship=0.5,car=0.5,oil_tank=0.1,wind_turbine=0.1,harbor=0.1 --name y26x-p2-f5-8-augbundle-e100
# y26x-p2 e100 augmentation bundle, seed 1
sbatch --mem=64GB --time=1-00:00:00 train.slurm yolo26.py --freeze 5-8 --epochs 100 --mult-noise 0.1 --gamma 0.2 --brightness-contrast 0.2 --elementwise --jpeg 50 --class-scale default=0.25,ship=0.5,car=0.5,oil_tank=0.1,wind_turbine=0.1,harbor=0.1 --seed 1 --name y26x-p2-f5-8-augbundle-s1-e100
# y26x-p2 e100 scale025 seed 1: seed noise of the baseline (per class), vs y26x-p2-f5-8-scale025-e100
sbatch --mem=64GB --time=1-00:00:00 train.slurm yolo26.py --freeze 5-8 --epochs 100 --mult-noise 0.1 --gamma 0.2 --brightness-contrast 0.2 --hyp scale=0.25 --seed 1 --name y26x-p2-f5-8-scale025-s1-e100
# y26x-p2 e100 mosaic off (Ultralytics mosaic 1.0 -> 0), vs y26x-p2-f5-8-scale025-e100
sbatch --mem=64GB --time=1-00:00:00 train.slurm yolo26.py --freeze 5-8 --epochs 100 --mult-noise 0.1 --gamma 0.2 --brightness-contrast 0.2 --hyp scale=0.25,mosaic=0 --name y26x-p2-f5-8-scale025-nomosaic-e100
# team08 repo (~/cs701-sar-team08, personal account), his best recipe --full: job 343955 (Blackwell)
sbatch --cpus-per-task=4 --mem=32G --gres=gpu:rtx_pro_6000_blackwell_max:1 train_personal.slurm --model yolo26x-p2 --freeze 5 6 7 8 --epochs 300 --scale 0 --mosaic 0 --speckle --cls-head --full --postprocess-from runs_yolo/yolo26x-p2_freeze5-6-7-8_scale0_mosaic0_speckle_clshead_ep300
# team08 two-stage stage 1: detection only (no --cls-head) --full, vs the base above
sbatch --cpus-per-task=4 --mem=32G --gres=gpu:l40s:1 train_personal.slurm --model yolo26x-p2 --freeze 5 6 7 8 --epochs 300 --scale 0 --mosaic 0 --speckle --full --postprocess-from runs_yolo/yolo26x-p2_freeze5-6-7-8_speckle_ep300
# team08 per-class scale (new --class-scale) --full, vs the base above
sbatch --cpus-per-task=4 --mem=32G --gres=gpu:l40s:1 train_personal.slurm --model yolo26x-p2 --freeze 5 6 7 8 --epochs 300 --scale 0.25 --class-scale ship=0.5,car=0.5,oil_tank=0.1,wind_turbine=0.1,harbor=0.1 --mosaic 0 --speckle --cls-head --full --postprocess-from runs_yolo/yolo26x-p2_freeze5-6-7-8_scale0_mosaic0_speckle_clshead_ep300
# y26x-p2 e100 bridge / playground x2 in the box classification BCE (--cls-weights), vs y26x-p2-f5-8-scale025-nomosaic-e100
sbatch --mem=64GB --time=1-00:00:00 train.slurm yolo26.py --freeze 5-8 --epochs 100 --mult-noise 0.1 --gamma 0.2 --brightness-contrast 0.2 --hyp scale=0.25,mosaic=0 --cls-weights bridge=2,playground=2 --name y26x-p2-f5-8-scale025-nomosaic-clsw2-e100
