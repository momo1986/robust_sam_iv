This repo is an enhanced repo of https://github.com/fudan-zvg/Semantic-Segment-Anything.

The new repo focuses on adversarial robustness of SAM on autonomous driving.

Running on cityscapes:
```bash

python scripts/main_ssa.py --ckpt_path ./ckp/sam_vit_h_4b8939.pth --save_img --world_size 1 --dataset cityscapes --data_dir data/cityscapes/leftImg8bit/val/ --gt_path data/cityscapes/gtFine/val/ --out_dir output_cityscapes_adversarial
python scripts/evaluation.py --gt_path data/cityscapes/gtFine/val/ --result_path output_cityscapes_adversarial/ --dataset cityscapes
```
Run on black-box attacks, please comment the code related to the white-box attacks and add corruptions in the pipeline:
```bash
pip install https://github.com/bethgelab/imagecorruptions
```

