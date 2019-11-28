# OCNet: Object Context Network for Scene Parsing (pytorch) 


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/object-contextual-representations-for/semantic-segmentation-on-coco-stuff-test)](https://paperswithcode.com/sota/semantic-segmentation-on-coco-stuff-test?p=object-contextual-representations-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/object-contextual-representations-for/semantic-segmentation-on-pascal-context)](https://paperswithcode.com/sota/semantic-segmentation-on-pascal-context?p=object-contextual-representations-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/object-contextual-representations-for/semantic-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k-val?p=object-contextual-representations-for)

	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/object-contextual-representations-for/semantic-segmentation-on-lip-val)](https://paperswithcode.com/sota/semantic-segmentation-on-lip-val?p=object-contextual-representations-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/object-contextual-representations-for/semantic-segmentation-on-cityscapes)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes?p=object-contextual-representations-for)


<h2>

	
```diff
+ Our work is submitted to the arXiv on Tue, 4 Sep 2018.
+ The concurrent work DANet is submitted to the arXiv on Sun, 9 Sep 2018.
+ We hope you can cite our works in your discussions of the related works if it helps.
```
</h2>

**Update @ 2019/11/28.**
We will release all of our [openseg.pytorch](https://github.com/PkuRainBow/openseg.pytorch) code within the future week to provide a strong baseline for the community, which supports all of our OCNet series.
Besides, we also provide the support of OCR on the [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1) code base.

**Update @ 2019/11/20.**
We have updated the paper [OCR](https://arxiv.org/abs/1909.11065), which is method of our **Rank#2** entry to the leaderboard of Cityscapes.
We could achieve **84.0%** with a simple novel post-processing scheme.

**Update @ 2019/07/31.**
We have released the paper [ISA](https://arxiv.org/abs/1907.12273), which is very easy to use and implement while being much more efficient than the conventional self-attention based OCNet or DANet.

**Update @ 2019/07/23.**
We achieve **Rank#1** on the leaderboard of Cityscapes (with a single model) and the results will be public soon.


**Update @ 2019/07/14.**
We achieve 83.3116%+ on the leaderboard of Cityscapes based on single model HRNet48V2 + OCR [training w/o any tricks]. More details will be shared in our new code base [openseg.pytorch](https://github.com/openseg-group/openseg.pytorch)



**Update @ 2019/05/27.**
We strongly recommend you to star our new code base [openseg.pytorch](https://github.com/openseg-group/openseg.pytorch) and our approach achieves SOTA on all 6 benchmarks and outperforms all the previous approaches by more than 1%~2%.




OCNet achieves 81.67 on the test set of Cityscapes with multi-scale method [0.75x, 1x, 1.25x].

OCNet achieves 45.45 on the validation set of ADE20K currently.

![Overall Framework of OCNet](OCNet.png?raw=true)

Please check the paper updated [OCNet](https://arxiv.org/pdf/1809.00916.pdf) here.


You are welcome to share our work with your friends. [zhihu share](https://zhuanlan.zhihu.com/p/43902175)

Please consider citing our work if you find it helps you,

```
@article{yuan2018ocnet,
  title={Ocnet: Object context network for scene parsing},
  author={Yuan Yuhui and Wang Jingdong},
  journal={arXiv preprint arXiv:1809.00916},
  year={2018}
}

@article{huang2019isa,
  title={Interlaced Sparse Self-Attention for Semantic Segmentation},
  author={Huang Lang and Yuan Yuhui and Guo Jianyuan and Zhang Chao and Chen Xilin and Wang Jingdong},
  journal={arXiv preprint arXiv:1907.12273},
  year={2019}
}

@article{yuan2019ocr,
  title={Object-Contextual Representations for Semantic Segmentation},
  author={Yuan Yuhui and Chen Xilin and Wang Jingdong},
  journal={arXiv preprint arXiv:1909.11065},
  year={2019}
}
```

## Introduction

Context is essential for various computer vision tasks.
The state-of-the-art scene parsing methods have exploited the effectiveness of the context defined over image-level.
Such context carries the mixture of objects belonging to different categories.

According to that the label of each pixel is defined as the category of the object it belongs to, we propose the Object Context that considers the objects belonging to the same category. 
The representation of any pixel P's object context is the aggregation of all the pixels' features that belong to the same category with P.
Since it is impractical to estimate all the objects belonging to the same category in advance,
we employ the self-attention method to approximate the objects by learning a pixel-wise similarity map.

We further propose the Pyramid Object Context and Atrous Spatial Pyramid Object Context to capture context of multiple scales.
Based on the object context, we introduce the OCNet and show that OCNet achieves state-of-the-art performance on both Cityscapes benchmark and ADE20K benchmark.


## Visualization of the learned Object Context
![Object Context learned with OCNet](OCNet_intro.jpg?raw=true)

## Experiment Results
All of our implementation is based on pytorch, OCNet can achieve competitive performance on various benchmarks such as Cityscapes and ADE20K without any bells and whistles.

The current performance on the Cityscapes test set of OCNet trained with only the fine-labeled set,

(**fix the bug of multi-scale testing**)


Method | Conference | Backbone | mIoU(\%) 
---- | --- | --- | --- 
RefineNet |  CVPR2017  | ResNet-101  |  73.6 
SAC  |  ICCV2017  | ResNet-101  |  78.1 
PSPNet |  CVPR2017  | ResNet-101  |  78.4
DUC-HDC | WACV2018 | ResNet-101 | 77.6 
AAF |   ECCV2018  | ResNet-101  |  77.1 
BiSeNet |   ECCV2018  | ResNet-101  |  78.9 
PSANet |  ECCV2018  | ResNet-101  |  80.1 
DFN  |  CVPR2018  | ResNet-101  |  79.3 
DSSPN | CVPR2018  | ResNet-101  | 77.8 
DenseASPP  |  CVPR2018  | DenseNet-161  |  80.6
**OCNet** | - |  ResNet-101 | **81.7**
**Fast-OCNet** | - |  ResNet-101 | **82.1**

The current performance on the ADE20K validation set of the OCNet


Method | Conference | Backbone | mIoU(\%) 
---- | --- | --- | --- 
RefineNet | CVPR2017  | ResNet-152  |  40.70  
PSPNet | CVPR2017  | ResNet-101  |  43.29 
SAC |  ICCV2017  |  ResNet-101  |  44.30  
PSANet  |  ECCV2018  |  ResNet-101  |  43.77  
EncNet |  CVPR2018  | ResNet-101  |  44.65
**OCNet** | - |  ResNet-101 | **45.45**


The current performance on the LIP validation set of the OCNet. We simply replace the PSP module with ASP-OC module within the [CE2P](https://github.com/liutinglt/CE2P) framework.

Method | Conference | Backbone | val mIoU(\%) 
---- | --- | --- | --- 
Attention+SSL |  CVPR2017  |  ResNet-101  |  44.73
SS-NAN    |  CVPR2017  |  ResNet-101  |  47.92 
MMAN    |  ECCV2018  |  ResNet-101  |  46.81
MuLA    |  ECCV2018  |  ResNet-101  |  49.30
JPPNet  |  PAMI2018  |  ResNet-101  |  51.37
CE2P    |  AAAI2019     | ResNet-101   |  53.10
**OCNet** | - |  ResNet-101 | **54.72**


## Enviroment
The code is developed using gcc5.4.0, python 3.6+, cuda8.0+ on Ubuntu 16.04. NVIDIA GPUs ared needed. The code is tested using 4 x NVIDIA P100 GPUS cards.
All the experiments on Cityscapes are tested on pytorch0.4.1. 

If you can only access TITAX Pascal / TITAN-1080Ti GPUs, you are recommended to modify the parameter **size** within the base-oc,

```
self.context = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            BaseOC_Module(in_channels=512, out_channels=512, key_channels=256, value_channels=256, 
            dropout=0.05, sizes=([2]))
            )
```

## Quick start


### Requirements
~~~~
torch=0.4.1
torchvision
tensorboardX
pillow
tqdm
h5py
scikit-learn
cv2
~~~~

Just install the offical pytorch with **pip install torch torchvision** is OK.

We mainly test all of our results based on the official pytorch 0.4.1.



### Train/Validate/Test the OCNet (Cityscapes)

We implement training, validating, testing in one script for convenience. You can achieve all the results by runing this script.

~~~~
sh run_asp_oc.sh
~~~~

You are expected to achieve ~79.5 mIoU on the validation set/ ~78.3 mIoU on the testing set (single scale).

You are also recommended to try the **run_base_oc.sh**, which can also achieve ~79.3 mIoU on the validation set / ~78.1 mIoU on the testing set (single scale).

To further improve the performance, you can employ the **CriterionOhemDSN_single** by setting that

~~~~
USE_OHEM=True
OHEMTHRES=0.7
OHEMKEEP=100000
~~~~

then you could expect to achieve ~80.4 mIoU on the validation set/ ~79.0 mIoU on the testing set (single scale).

To achieve the 81.2 on the testing set, you need to train the model with both the training set and validation set for 80,000 iterations first(achieve ~80.5 on the testing set(multiple-scale + flip)), then you need to finetune this model for 100,000 iterations with fixed learning rate(1e-4). We adopt the online hard example mining accordingly.


## Data preparation

For the cityscapes dataset, please download the dataset from the Cityscapes webset. Unzip all the images under the path "./OCNet/dataset/cityscapes". Ensure the path tree like below within the folder "./OCNet/dataset/cityscapes".

```
|-- README
|-- get_cs_extra.sh
|-- gtCoarse
|   |-- README
|   |-- license.txt
|   |-- train
|   |-- train_extra
|   `-- val
|-- gtFine
|   |-- README
|   |-- license.txt
|   |-- test
|   |-- train
|   `-- val
|-- leftImg8bit
|   |-- README
|   |-- license.txt
|   |-- test
|   |-- train
|   |-- train_extra
|   `-- val
|-- license.txt
`-- tree.txt
```
## Pretrained Models

Please put the pretrained models under the folder "./OCNet/pretrained_model"

[ImageNet Pretrained ResNet-101](http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth)

## Docker with the enviroments

[rainbowsecret/pytorch04:20180719](https://hub.docker.com/r/rainbowsecret/pytorch04/tags/)

## Other problems (Performance gap between the Validation set and Testing set)
We find that the mIoU of the class train is unstable sometimes. For example, we run our code for 5 times, there can exist one time the mIoU for class train is 0.42 while we can get 0.75 for other 4 times.

There also exist some problems about the validation/testing set accuracy gap.
For example, if you run the base-oc method for two times, you can achieve 79.3 and 79.8 mIou on the validation set separately while the testing mIou can be 78.55 and 77.69.
Thus I recommend to you to run our methods for multiple times if you want to achieve good performance on the testing set while our method performs pretty robust on the validation set as the reason of the distribution gaps between the training/validation set and the testing set.

## Acknowledgment
This project is created based on the reproduced Deeplabv3, PSPNet (pytorch) provided by [Zilong Huang](https://github.com/speedinghzl) and he retains all the copyright of the reproduced Deeplabv3, PSPNet related code. 

## Thanks to the Third Party Libs
[InplaceABN](https://github.com/mapillary/inplace_abn)

[Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch).

[Pytorch-Deeplab](https://github.com/speedinghzl/Pytorch-Deeplab)

[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)

[semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch)


