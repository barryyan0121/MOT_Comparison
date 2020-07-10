# 多目标追踪学习报告
## MOT介绍
目标跟踪是机器视觉中一类被广为研究的重要问题，分为单目标跟踪与多目标跟踪。前者跟踪视频画面中的单个目标，后者则同时跟踪视频画面中的多个目标，得到这些目标的运动轨迹。

基于视觉的目标自动跟踪在智能监控、动作与行为分析、自动驾驶等领域都有重要的应用。例如，在自动驾驶系统中，目标跟踪算法要对运动的车、行人、其他动物的运动进行跟踪，对它们在未来的位置、速度等信息作出预判。

目标跟踪算法可以进行轨迹特征的自动分析和提取，以弥补视觉目标检测的不足，有效的去除错误的检测，增加遗漏的检测，为进一步的行为分析提供基础。

基于深度学习的算法在图像和视频识别任务中取得了广泛的应用和突破性的进展。从图像分类问题到行人重识别问题，深度学习方法相比传统方法表现出极大的优势。与行人重识别问题紧密相关的是行人的多目标跟踪问题。

在多目标跟踪(MOT)问题中，算法需要根据每一帧图像中目标的检测结果，匹配已有的目标轨迹；对于新出现的目标，需要生成新的目标；对于已经离开摄像机视野的目标，需要终止轨迹的跟踪。这一过程中，目标与检测的匹配可以看作为目标的重识别，例如，当跟踪多个行人时，把已有的轨迹的行人图像集合看作为图像库(gallery)，而检测图像看作为查询图像(query)，检测与轨迹的匹配关联过程可以看作由查询图像检索图像库的过程。

与传统的行人重识别(Re-ID)不同的是，行人多目标跟踪中的检测与行人轨迹的匹配关联问题更加复杂，具体表现在下面三个方面：首先，多目标跟踪中的目标轨迹是频繁发生变化的，图像样本库的数量和种类并不固定。其次，检测结果中可能出现新的目标，也可能不包括已有的目标轨迹。另外，检测图像并不像传统行人重识别中的查询图像都是比较准确的检测结果，通常，行人多目标跟踪场景下的检测结果混杂了一些错误的检测(false-alarms)，而由于背景以及目标之间的交互，跟踪中的行人检测可能出现图像不对齐、多个检测对应同一目标、以及一个检测覆盖了多个目标这些情况。

本文将探讨DeepSORT算法和FairMOT算法，以及目标检测(Object Detection)和行人重识别(Re-ID)之间的联系。

## MOT算法的通常工作流程
(1)给定视频的原始帧<br>
(2)运行对象检测器以获得对象的边界框<br>
(3)对于每个检测到的物体计算出不同的特征，通常是视觉和运动特征<br>
(4)之后，相似度计算步骤计算两个对象属于同一目标的概率<br>
(5)最后，关联步骤为每个对象分配数字ID

## DeepSORT(SIMPLE ONLINE AND REALTIME TRACKING WITH A DEEP ASSOCIATION METRIC)
### 导语
DeepSort是在Sort目标追踪基础上的改进。引入了在行人重识别数据集上离线训练的深度学习模型，在实时目标追踪过程中，提取目标的表观特征进行最近邻匹配，可以改善有遮挡情况下的目标追踪效果。同时，也减少了目标ID跳变的问题。算法的核心思想还是用一个传统的单假设追踪方法，方法使用了递归的卡尔曼滤波和逐帧的数据关联。为了学习DeepSORT，我们首先需要了解SORT原理。
### SORT(SIMPLE ONLINE AND REALTIME TRACKING)原理
SORT作为一个粗略的框架，核心就是两个算法：**卡尔曼滤波**和**匈牙利匹配**。

**卡尔曼滤波**分为两个过程：预测(predict)和更新(update)。预测过程：当一个小车经过移动后，且其初始定位和移动过程都是高斯分布时，则最终估计位置分布会更分散，即更不准确；更新过程：当一个小车经过传感器观测定位，且其初始定位和观测都是高斯分布时，则观测后的位置分布会更集中，即更准确。

**匈牙利算法**解决的是一个分配问题。SK-learn库的linear_assignment和scipy库的linear_sum_assignment都实现了这一算法，只需要输入cost_matrix即代价矩阵就能得到最优匹配。不过要注意的是这两个库函数虽然算法一样,但给的输出格式不同。此代码使用的是SK-learn库的linear_assignment。

DeepSORT的优化主要就是基于匈牙利算法里的这个代价矩阵。它在IOU Matching之前做了一次额外的级联匹配，利用了外观特征和马氏距离。

SORT流程整体可以拆分为两个部分，分别是匹配过程和卡尔曼预测加更新过程。

关键步骤：轨迹卡尔曼滤波**预测** → 使用**匈牙利算法**将预测后的tracks和当前帧中的detecions进行匹配(**IOU匹配**) → 卡尔曼滤波**更新**

### DeepSORT原理
DeepSORT算法和SORT基本一样，就多了**级联匹配**(Matching Cascade)和新轨迹的确认(confirmed)。

DeepSORT对每一帧的处理流程如下：

检测器得到bbox → 生成detections → 卡尔曼滤波预测→ 使用**匈牙利算法**将预测后的tracks和当前帧中的detections进行匹配(**级联匹配**和**IOU匹配**) → 卡尔曼滤波**更新**

关键步骤：轨迹卡尔曼滤波预测 → 使用**匈牙利算法**将预测后的tracks和当前帧中的detections进行匹配(**级联匹配**和**IOU匹配**) → 卡尔曼滤波**更新**

Frame 0：检测器检测到了3个detections，当前没有任何tracks，将这3个detections初始化为tracks
Frame 1：检测器又检测到了3个detections，对于Frame 0中的tracks，先进行预测得到新的tracks，然后使用匈牙利算法将新的tracks与detections进行匹配，得到(track, detection)匹配对，最后用每对中的detection更新对应的track

**级联匹配**(Matching Cascade)是核心，DeepSORT的绝大多数创新点都在这里面。
<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">
关于为什么新轨迹要连续三帧命中才确认？个人认为有这样严格的条件和测试集有关系。因为测试集给的检测输入非常的差，误检有很多，因此轨迹的产生必须要更严格的条件。

### 代码解读
#### 检测
使用Yolo作为检测器，检测当前帧中的bbox
#### 生成detections
将检测到的bbox转换成detections
#### 卡尔曼滤波预测阶段
使用卡尔曼滤波预测前一帧中的tracks在当前帧的状态
#### 匹配
首先对基于外观信息的马氏距离(Mahalanobis distance)计算tracks和detections的代价矩阵，然后相继进行**级联匹配**和**IOU匹配**，最后得到当前帧的所有匹配对、未匹配的tracks以及未匹配的detections
#### 卡尔曼滤波更新阶段
对于每个匹配成功的track，用其对应的detection进行更新，并处理未匹配tracks和detections

### 运行结果
生成视频帧数 处理速度 结果为文本文档

## FairMOT (A simple baseline for one-shot Multi-Object Tracking)

## 目标检测(Object Detection)
### YOLO
### EfficientNet/EfficientDet

## 行人重识别(Re-ID)

## Acknowledgement
https://zhuanlan.zhihu.com/p/59148865<br>
https://zhuanlan.zhihu.com/p/90835266<br>
https://zhuanlan.zhihu.com/p/80764724<br>
https://zhuanlan.zhihu.com/p/114349651<br>
https://www.cnblogs.com/yanwei-li/p/8643446.html<br>
https://blog.csdn.net/cdknight_happy/article/details/79731981
