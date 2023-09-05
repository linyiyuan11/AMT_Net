# AMTNet
The pytorch implementation for AMTNet in paper ["An attention-based multiscale transformer network for remote sensing image
change detection"](https://www.sciencedirect.com/science/article/abs/pii/S092427162300182X?CMX_ID=&SIS_ID=&dgcid=STMJ_AUTH_SERV_PUBLISHED&utm_acid=276849605&utm_campaign=STMJ_AUTH_SERV_PUBLISHED&utm_in=DM391842&utm_medium=email&utm_source=AC_)on ["ISPRS Journal of Photogrammetry and Remote Sensing"](https://www.sciencedirect.com/journal/isprs-journal-of-photogrammetry-and-remote-sensing).
# Requirements
* Python 3.9
* Pytorch 1.12 
# DataSet
* Download the [CLCD Dataset](https://pan.baidu.com/share/init?surl=Un-bVxUm1N9IHiDOXLLHlg&pwd=miu2)
* Download the [HRSCD Dataset](https://ieee-dataport.org/open-access/hrscd-high-resolution-semantic-change-detection-dataset)
* Download the [WHU-CD Dataset](http://gpcv.whu.edu.cn/data/building_dataset.html)
* Download the [LEVIR-CD Dataset](http://chenhao.in/LEVIR/)
```
Prepare datasets into following structure and set their path in train_options.py
├─Train
│  ├─time1
│  │─time2
│  │─label
├─Test
│  ├─time1
│  │─time2
│  │─label
```
# Train
```
python train.py
```
All the hyperparameters can be adjusted in train_options.py
# model zool
The models with the scores can be downloaded from[Baidu Cloud]().
# Acknowledgments
This code is heavily borrowed from [MSCANet](https://github.com/liumency/CropLand-CD) and [changer](https://github.com/likyoo/open-cd/tree/main).
# Citation
If you find this repo useful for your research, please consider citing the paper as follows:
```
@article{liu2023attention,
  title={An attention-based multiscale transformer network for remote sensing image change detection},
  author={Liu, Wei and Lin, Yiyuan and Liu, Weijia and Yu, Yongtao and Li, Jonathan},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={202},
  pages={599--609},
  year={2023},
  publisher={Elsevier}
}
```
 
