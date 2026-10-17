# sbatch commands

```bash
# FGFM on levels 0-2, 24 ep vs swin_b-moelora (L40S: level 0 likely does not fit on a 3090)
sbatch --gres=gpu:l40s:1 train.slurm --backbone swin_b --init pretrained --adapt moelora --fgfm-levels 0-2 --name swin_b-moelora-fgfm
```
