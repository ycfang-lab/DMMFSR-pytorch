##DMMF Super-resolution
这是一个的超分辨率的工作，在本工作中，我们创造性地提出了一个动态多任务和相互特征融合的超分网络(DMMFSR)，该网络能够在保持图像更多边缘结构的同时缓解解构畸变问题。具体来说，我们利用动态多任务同时学习巡林低频和高频图像信息提取网络，让两个网络相互促进，以此提高各个任务的性能。

### Prerequisites:
1. Python 3.6

2. PyTorch = 0.4.0

3. numpy

4. skimage

5. imageio

6. matplotlib

7. tqdm

## Dataset
We use DIV2K dataset to train our model. Please download it from <a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">here</a>  or  <a href="https://cv.snu.ac.kr/research/EDSR/DIV2K.tar">SNU_CVLab</a>.

Extract the file and put it into the Train/dataset.

Only DIV2K is used as the training dataset !!!

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