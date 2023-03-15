# DMMFSR-pytorch
An implementation of single image super-resolution via dynamic multi-task learning and mutual multi-level feature fusion.

In this work, we creatively propose a single image super-resolution network via dynamic multi-task learning and mutual multi-level feature fusion (DMMFSR), which can ease the deconstruction and distortion while maintaining a more edge structure of the image. Specifically, we use dynamic multi-task learning to simultaneously learn low-frequency and high-frequency image information extraction networks and let the two networks promote each other to improve the performance of each task.

## Code and pretrained models
Our DMMFSR is qualitatively analyzed with SRCNN, EDSR, MSRN, and SeaNet on Set5, Set14, BSDS100, and Urban100 datasets. We provide the training code and pre-training models of DMMFSR under the DMMFSR folder. In addition, the code and pre-training model used to obtain the SRCNN, EDSR, MSRN, and SeaNet models on each data set are also in the corresponding folders.

The official codes of SRCNN, EDSR, MSRN, and SeaNet can be found here:

[SRCNN - Dong et al. PAMI 2015.](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)

[EDSR - Lim et al. CVPRW 2017.](https://github.com/sanghyun-son/EDSR-PyTorch)

[MSRN - Li et al. ECCV 2018.](https://github.com/MIVRC/MSRN-PyTorch)

[SeaNet - Fang et al. TIP 2020](https://github.com/MIVRC/SeaNet-PyTorch)

## Prerequisites:
1. Python 3.6

2. PyTorch = 0.4.0

3. numpy

4. skimage

5. imageio

6. matplotlib

7. tqdm


## Datasets
We use DIV2K dataset to train our model. Please download it from <a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">here</a>  or  <a href="https://cv.snu.ac.kr/research/EDSR/DIV2K.tar">SNU_CVLab</a>.

Extract the file and put it into the Train/dataset.

Only DIV2K is used as the training dataset !!!

The datasets for qualitative comparisons can be download from here:

[Set5 - Bevilacqua et al. BMVC 2012.](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)

[Set14 - Zeyde et al. LNCS 2010.](https://sites.google.com/site/romanzeyde/research-interests)

[B100 - Martin et al. ICCV 2001.](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)

[Urban100 - Huang et al. CVPR 2015.](https://huggingface.co/datasets/eugenesiow/Urban100)

## Train
修改option.py文件中的--data_train参数

修改option.py文件中的--data_test参数

修改option.py文件中的--data_range参数

```shell
# DMMFSR x2  LR: 48 * 48  HR: 96 * 96
python main.py --template DMMFSR --save DMMFSR_X2 --scale 2 --reset --save_results --patch_size 96 --ext sep_reset

# DMMFSR x3  LR: 48 * 48  HR: 144 * 144

python main.py --template SEDMMFSRAN --save DMMFSR_X3 --scale 3 --reset --save_results --patch_size 144 --ext sep_reset

# DMMFSR x4  LR: 48 * 48  HR: 192 * 192

python main.py --template DMMFSR --save DMMFSR_X4 --scale 4 --reset --save_results --patch_size 192 --ext sep_reset
```

## Test
Test代码可以参考trainer中的test函数
