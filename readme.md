# A Coarse Semantics-guided Multi-scale Interaction Network for Camouflaged Object Detection

> 

## 1. Preface

*   

## 2. Overview

### 2.1. Introduction


Camouflaged Object Detection (COD) seeks to identify and segment camouflaged targets embedded in complex environments, mainly challenge due to the intrinsic similarity between foreground objects and their background. Although context-fusion networks have moderately enhanced detection performance, the loss of fine-grained details and the inadequacy of coarse map guidance frequently result in incomplete targets and blurred boundaries. To address these issues, we propose a coarse semantics-guided multi-scale interaction network (CMNet). First, our approach employs a dual-branch architecture: one branch uses adaptive pooling and convolution to capture scene-level semantic cues, while the other exploits channel splitting and convolutions to augment sensitivity to local textural variations. Then, an aggregation module progressively fuses multi-scale features to generate a coarse prior that delineates the overall target location, and acquiring a coarse attention map via an attention module to guide contextual secondary aggregation. The results of experiment on four different datasets demonstrate that CMNet achieves state-of-the-art performance, underscoring its efficacy in camouflage object detection and its promising potential for practical applications.** 

### 2.2. Framework Overview

![](.\fig\f1_1_1_1.png)

***Figure1:*** Our model structure, including the main components, i.e., Dual-Branch Feature Enhancement Module (DFEM) to enhance feature expression, Coarse
Semantics Guide Module (CSGM) injecting coarse information into feature and Secondary Context Aggregation module (SCAM) using to re-fuse feature.
See Sec. 3 for details.

### 2.3. Qualitative Results

![](.\fig\cod.png)

***Figure2:*** Qualitative results of our proposed  model and some state-of-the-art COD methods. T

## 3. Proposed Method

### 3.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with one NVIDIA 2080Ti GPU of 32 GB Memory.

1.  Configuring your environment (Prerequisites):

    *   Installing necessary packages:&#x20;

        python 3.6&#x20;

        torch 1.11.0

        numpy 1.22.4

        mmcv-full 1.7.1

        timm 0.6.13

        mmdet 2.19.1

2. Downloading necessary data:

   *   downloading dataset and move it into `./data/`, which can be found from [Baidu Drive](https://pan.baidu.com/s/15ro0EjyKKqPLRFVs8g865w?pwd=sm2e).
   *   downloading our weights and move it into `./checkpoint/Net_best.pth`,  which can be found from [Baidu Drive](https://pan.baidu.com/s/1ZI-5N0ZbhHX-S5fl2IQfkQ?pwd=cmne ).提取码: cmne&#x20;
   *   downloading PVTv2-Large weights and move it into `models/pvt_v2_b2.pth`, which can be found from [Baidu Drive](https://pan.baidu.com/s/1ZI-5N0ZbhHX-S5fl2IQfkQ?pwd=cmne ).提取码: cmne&#x20;

3. Training Configuration:

   *   FOR COD : After you download training dataset, just run `train_oc.py` to train our model.
   *   FOR SOD ： just run `train_sod.py` to train our model.

4.  Testing Configuration:

    *   After you download all the pre-trained model and testing dataset, just run `test_oc.py` for cod to generate the final prediction maps, and `test_sod.py` for sod 

    * You can also download prediction sod maps from [Baidu Drive](https://pan.baidu.com/s/1GwCYNd_AR8IB-SBq0TNQlg?pwd=cmne ). 提取码: cmne
    
    * cod maps [Baidu Drive](https://pan.baidu.com/s/1p6Qz_3Ng7V6SY4zpx-H98Q?pwd=cmne ). 提取码: cmne 
    
      

### 3.2 Evaluating your trained model:

One evaluation is written in Python code  please follow this the instructions in `./eval.py` and just run it to generate the evaluation results in. We implement four metrics: MAE (Mean Absolute Error), weighted F-measure, mean E-measure, S-Measure.

**[⬆ back to top](#1-preface)**
