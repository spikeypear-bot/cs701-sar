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
```
