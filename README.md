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
![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/deepsort/images/SORT.jpg)<br>
SORT作为一个粗略的框架，核心就是两个算法：**卡尔曼滤波**和**匈牙利匹配**。

**卡尔曼滤波**分为两个过程：预测(predict)和更新(update)。预测过程：当一个小车经过移动后，且其初始定位和移动过程都是高斯分布时，则最终估计位置分布会更分散，即更不准确；更新过程：当一个小车经过传感器观测定位，且其初始定位和观测都是高斯分布时，则观测后的位置分布会更集中，即更准确。

**匈牙利算法**解决的是一个分配问题。SK-learn库的linear_assignment和scipy库的linear_sum_assignment都实现了这一算法，只需要输入cost_matrix即代价矩阵就能得到最优匹配。不过要注意的是这两个库函数虽然算法一样,但给的输出格式不同。此代码使用的是SK-learn库的linear_assignment。

SORT流程整体可以拆分为两个部分，分别是匹配过程和卡尔曼预测加更新过程。

关键步骤：轨迹卡尔曼滤波**预测** → 使用**匈牙利算法**将预测后的tracks和当前帧中的detecions进行匹配(**IOU匹配**) → 卡尔曼滤波**更新**

### DeepSORT原理
![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/deepsort/images/DeepSORT.jpg)<br>
DeepSORT算法和SORT基本一样，就多了**级联匹配**(Matching Cascade)和新轨迹的确认(confirmed)。

DeepSORT对每一帧的处理流程如下：

检测器得到bbox → 生成detections → 卡尔曼滤波预测→ 使用**匈牙利算法**将预测后的tracks和当前帧中的detections进行匹配(**级联匹配**和**IOU匹配**) → 卡尔曼滤波**更新**

关键步骤：轨迹卡尔曼滤波预测 → 使用**匈牙利算法**将预测后的tracks和当前帧中的detections进行匹配(**级联匹配**和**IOU匹配**) → 卡尔曼滤波**更新**

传统的解决检测结果与追踪预测结果的关联的方法是使用**匈牙利算法**。本文作者同时考虑了运动信息的关联和目标外观信息的关联。
DeepSORT的优化主要就是基于**匈牙利算法**里的这个代价矩阵。它在IOU Matching之前做了一次额外的级联匹配，利用了外观特征和**马氏距离**。
运动信息的关联：使用了对已存在的运动目标的运动状态的kalman预测结果与检测结果之间的马氏距离进行运行信息的关联。<br>
<img src="https://render.githubusercontent.com/render/math?math={d_{i,j}^{(1)}} = \left ( d_{j} - y_{i} \right )^{T} S_{i}^{-1}\left ( d_{j} - y_{i} \right )">，<br><img src="https://render.githubusercontent.com/render/math?math=d_{j}">表示第j个检测框的位置，<img src="https://render.githubusercontent.com/render/math?math=y_{i}">表示第i个追踪器对目标的预测位置，<img src="https://render.githubusercontent.com/render/math?math=S_{i}">表示检测位置与平均追踪位置之间的协方差矩阵。马氏距离通过计算检测位置和平均追踪位置之间的标准差将状态测量的不确定性进行了考虑。<br>
如果某次关联的马氏距离小于指定的阈值<img src="https://render.githubusercontent.com/render/math?math=t^{(1)}">，则设置运动状态的关联成功。使用的函数为<br><img src="https://render.githubusercontent.com/render/math?math={b_{i,j}^{(1)}} = \mathbb{I}\left [ {d_{i,j}^{(1)}} \leqslant t^{(1)} \right ]">，作者设置<img src="https://render.githubusercontent.com/render/math?math=t^{(1)}=9.4877">。

目标外观信息的关联：当运动的不确定性很低的时候，上述的马氏距离匹配是一个合适的关联度量方法，但是在图像空间中使用卡尔曼滤波进行运动状态估计只是一个比较粗糙的预测。特别是相机存在运动时会使得马氏距离的关联方法失效，造成出现ID switch的现象。因此引入了第二种关联方法，对每一个的检测块<img src="https://render.githubusercontent.com/render/math?math=d_{j}">求一个特征向量<img src="https://render.githubusercontent.com/render/math?math=r_{i}">，限制条件是<img src="https://render.githubusercontent.com/render/math?math=\left \| r_{i} \right \| = 1">。作者对每一个追踪目标构建一个gallary，存储每一个追踪目标成功关联的最近100帧的特征向量。那么第二种度量方式就是计算第i个追踪器的最近100个成功关联的特征集与当前帧第j个检测结果的特征向量间的最小余弦距离。计算公式为：<br><img src="https://render.githubusercontent.com/render/math?math=d^{(2)}(i,j)=min\left \{ r_{j}^{T}r_{k}^{(i)} | r_{k}^{(i)}\in R_{i}\right \}"><br>
如果上面的距离小于指定的阈值，那么这个关联就是成功的。阈值是从单独的训练集里得到的。

使用两种度量方式的线性加权作为最终的度量，<br><img src="https://render.githubusercontent.com/render/math?math=c_{i,j} = \lambda d^{(1)}(i,j) + (1-\lambda)d^{(2)}(i,j)">，只有<img src="https://render.githubusercontent.com/render/math?math=c_{i,j}">位于两种度量阈值的交集时，才认为实现了正确的关联。<br>
距离度量对短期的预测和匹配效果很好，但对于长时间的遮挡的情况，使用外观特征的度量比较有效。对于存在相机运动的情况，可以设置<img src="https://render.githubusercontent.com/render/math?math=\lambda = 0">。



**级联匹配**(Matching Cascade)是核心，DeepSORT的绝大多数创新点都在这里面。
![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/deepsort/images/matching%20cascade%20step.png)

当一个目标长时间被遮挡之后，卡尔曼滤波预测的不确定性就会大大增加，状态空间内的可观察性就会大大降低。假如此时两个追踪器竞争同一个检测结果的匹配权，往往遮挡时间较长的那条轨迹因为长时间未更新位置信息，追踪预测位置的不确定性更大，即协方差会更大，**马氏距离**计算时使用了协方差的倒数，因此**马氏距离**会更小，因此使得检测结果更可能和遮挡时间较长的那条轨迹相关联，这种不理想的效果往往会破坏追踪的持续性。因为相机抖动明显，卡尔曼预测所基于的匀速运动模型并不准确，所以**马氏距离**其实并没有什么作用，主要是通过阈值矩阵(Gate Matrix)对代价矩阵(Cost Matrix)做了一次阈值限制。

级联匹配的核心思想就是由小到大对消失时间相同的轨迹进行匹配，这样首先保证了对最近出现的目标赋予最大的优先权，也解决了上面所述的问题。

![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/deepsort/images/matching_cascade.jpg)

**级联匹配**流程图里上半部分就是特征提取和相似度估计，也就是算这个分配问题的代价函数。主要由两部分组成：代表运动模型的**马氏距离**和代表外观模型的**Re-ID**特征。

**级联匹配**流程图里下半部分数据关联作为流程的主体。为什么叫级联匹配，主要是它的匹配过程是一个循环。从missing age = 0的轨迹（即每一帧都匹配上，没有丢失过的）到missing age = 30的轨迹（即丢失轨迹的最大时间30帧）挨个的和检测结果进行匹配。也就是说，对于没有丢失过的轨迹赋予优先匹配的权利，而丢失的最久的轨迹最后匹配。

在匹配的最后阶段还对unconfirmed和age=1的未匹配轨迹和检测目标进行基于IoU的匹配。这可以缓解因为表观突变或者部分遮挡导致的较大变化。当然有好处就有坏处，这样做也有可能导致一些新产生的轨迹被连接到了一些旧的轨迹上。但这种情况较少。

工作流程示例：
```
Frame 0：检测器检测到了3个detections，当前没有任何tracks，将这3个detections初始化为tracks。

Frame 1：检测器又检测到了3个detections，对于Frame 0中的tracks，先进行预测得到新的tracks，然后使用匈牙利算法将新的tracks与detections进行匹配，得到(track, detection)匹配对，最后用每对中的detection更新对应的track。
```

**深度特征描述器**<br>
网络结构：<br>
![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/deepsort/images/cnn.png)<br>
在行人重识别数据集上离线训练残差网络模型。输出128维的归一化的特征。

### 代码解读
论文中提供的代码是如下地址: https://github.com/nwojke/deep_sort

按视频帧顺序处理，每一帧的处理流程如下:

![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/object%20detection/demo/cri2dh3p5w.png)

#### 检测并生成detections
读取当前帧目标检测框的位置及各检测框图像块的深度特征(此处在处理实际使用时需要自己来提取)

```python
# deep_sort_app.py
def create_detections(detection_mat, frame_idx, min_height=0):
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list
```
detection_mat : 格式为ndarray的检测矩阵。该矩阵的前十行均为标准MOTChallenge检测格式，剩余列项存储着每个检测目标的特征向量。<br>
frame_idx : 格式为int的帧数索引。<br>
min_height : 格式为int的最小检测边界框高度。比该值小的检测数据会被丢弃。

根据置信度对检测框进行过滤，即对置信度不足够高的检测框及特征予以删除
```python
# deep_sort_app.py
# 加载图像并生成detections
detections = create_detections(
seq_info["detections"], frame_idx, min_detection_height)
detections = [d for d in detections if d.confidence >= min_confidence]
```

Detection类用于保存通过目标检测器得到的一个检测框，包含top left坐标+框的宽和高，以及该bbox的置信度还有通过Re-ID获取得到的对应的embedding。除此以外提供了不同bbox位置格式的转换方法：
* tlwh: 代表左上角坐标+宽高
* tlbr: 代表左上角坐标+右下角坐标
* xyah: 代表中心坐标+宽高比+高

对检测框进行非最大值抑制，消除一个目标身上多个框的情况
```python
# deep_sort_app.py
# 运行非最大值抑制
boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
```
#### 卡尔曼滤波预测阶段
使用卡尔曼滤波预测前一帧中的tracks在当前帧的状态

```python
# deep_sort_app.py
tracker.predict()
```

```python
# tracker.py
# 向前进一个时间步长传播轨道状态分布
# 这个函数应该在卡尔曼滤波更新之前每个时间点调用一次
def predict(self):
    for track in self.tracks:
        track.predict(self.kf)
```

预测完之后，需要对每一个tracker的self.time_since_update += 1。

```python
# track.py
class Track:
    # 一个轨迹的信息，包含(x,y,a,h) & v
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.
    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        # max age是一个存活期限，默认为70帧,在
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1 
        # hits和n_init进行比较
        # hits每次update的时候进行一次更新（只有match的时候才进行update）
        # hits代表匹配上了多少次，匹配次数超过n_init就会设置为confirmed状态
        self.age = 1 # 没有用到，和time_since_update功能重复
        self.time_since_update = 0
        # 每次调用predict函数的时候就会+1
        # 每次调用update函数的时候就会设置为0

        self.state = TrackState.Tentative
        self.features = []
        # 每个track对应多个features, 每次更新都将最新的feature添加到列表中
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init  # 如果连续n_init帧都没有出现失配，设置为deleted状态
        self._max_age = max_age  # 上限
```

Track类主要存储的是轨迹信息，mean和covariance是保存的框的位置和速度信息，track_id代表分配给这个轨迹的ID。state代表框的状态，有三种：
* Tentative: 不确定态，这种状态会在初始化一个Track的时候分配，并且只有在连续匹配上n_init帧才会转变为确定态。如果在处于不确定态的情况下没有匹配上任何detection，那将转变为删除态。
* Confirmed: 确定态，代表该Track确实处于匹配状态。如果当前Track属于确定态，但是失配连续达到max age次数的时候，就会被转变为删除态。
* Deleted: 删除态，说明该Track已经失效。

```python
# track.py
# 使用卡尔曼滤波器预测步骤
def predict(self, kf):
    self.mean, self.covariance = kf.predict(self.mean, self.covariance)
    self.age += 1
    self.time_since_update += 1
```
kf : 为卡尔曼滤波器(kalman_filter.KalmanFilter)

执行kalman滤波公式1和2:<img src="https://render.githubusercontent.com/render/math?math=x(k)=Ax(k-1)">和<img src="https://render.githubusercontent.com/render/math?math=p(k)=Ap(k-1)A^{T}+Q">,其中，<img src="https://render.githubusercontent.com/render/math?math=x(k-1)">为目标的状态信息(代码中的mean)，<img src="https://render.githubusercontent.com/render/math?math=p(k-1)">为目标的估计误差(代码中的covariance)，A为状态转移矩阵，Q为系统误差。

```python
# kalman_filter.py
def predict(self, mean, covariance):
# 运行卡尔曼滤波器预测步骤
    # 相当于得到t时刻估计值
    # Q 预测过程中噪声协方差
    std_pos = [
        self._std_weight_position * mean[3],
        self._std_weight_position * mean[3],
        1e-2,
        self._std_weight_position * mean[3]]
    std_vel = [
        self._std_weight_velocity * mean[3],
        self._std_weight_velocity * mean[3],
        1e-5,
        self._std_weight_velocity * mean[3]]
    # np.r_ 按列连接两个矩阵
    # 初始化噪声矩阵Q
    motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
    # 卡尔曼滤波公式1(x' = Fx)
    mean = np.dot(self._motion_mat, mean)
    
    # 卡尔曼滤波公式2(P' = FPF^T+Q)
    covariance = np.linalg.multi_dot((
        self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

    return mean, covariance
```

mean : 格式为ndarray的位于前一个时间点的目标状态八维向量   
covariance : 格式为ndarray的位于前一个时间点的目标状态8x8的协方差矩阵
输出格式为(ndarray, ndarray)的预测目标平均向量和协方差矩阵，未被观测的速度将被初始化为0

#### ReID特征提取部分
ReID网络是独立于目标检测和跟踪器的模块，功能是提取对应bounding box中的feature,得到一个固定维度的embedding作为该bbox的代表，供计算相似度时使用。
```python
# Extractor.py
class Extractor(object):
    def __init__(self, model_name, model_path, use_cuda=True):
        self.net = build_model(name=model_name,
                               num_classes=96)
        self.device = "cuda" if torch.cuda.is_available(
        ) and use_cuda else "cpu"
        state_dict = torch.load(model_path)['net_dict']
        self.net.load_state_dict(state_dict)
        print("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (128,128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.3568, 0.3141, 0.2781],
                                 [0.1752, 0.1857, 0.1879])
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([
            self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops
        ],dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()
```
模型训练是按照传统ReID的方法进行，使用Extractor类的时候输入为一个list的图片，得到图片对应的特征。

#### 匹配
首先对基于外观信息的马氏距离(Mahalanobis distance)计算跟踪框(tracks)和检测框(detections)的代价矩阵，然后相继进行**级联匹配**和**IOU匹配**，最后得到当前帧的所有匹配对、未匹配的tracks以及未匹配的detections。

```python
# deep_sort_app.py
tracker.update(detections)
```

Tracker类是最核心的类，Tracker中保存了所有的轨迹信息，负责初始化第一帧的轨迹、卡尔曼滤波的预测和更新、负责级联匹配、IOU匹配等等核心工作。

update函数
```python
# tracker.py
def update(self, detections):
    # 进行测量的更新和轨迹管理
    """Perform measurement update and track management.

    Parameters
    ----------
    detections : List[deep_sort.detection.Detection]
        A list of detections at the current time step.

    """
    # Run matching cascade.
    matches, unmatched_tracks, unmatched_detections = \
        self._match(detections)

    # Update track set.
    # 1. 针对匹配上的结果
    for track_idx, detection_idx in matches:
        # track更新对应的detection
        self.tracks[track_idx].update(self.kf, detections[detection_idx])

    # 2. 针对未匹配的tracker,调用mark_missed标记
    # track失配，若待定则删除，若update时间很久也删除
    # max age是一个存活期限，默认为70帧
    for track_idx in unmatched_tracks:
        self.tracks[track_idx].mark_missed()

    # 3. 针对未匹配的detection， detection失配，进行初始化
    for detection_idx in unmatched_detections:
        self._initiate_track(detections[detection_idx])

    # 得到最新的tracks列表，保存的是标记为confirmed和Tentative的track
    self.tracks = [t for t in self.tracks if not t.is_deleted()]

    # Update distance metric.
    active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
    # 获取所有confirmed状态的track id
    features, targets = [], []
    for track in self.tracks:
        if not track.is_confirmed():
            continue
        features += track.features  # 将tracks列表拼接到features列表
        # 获取每个feature对应的track id
        targets += [track.track_id for _ in track.features]
        track.features = []

    # 距离度量中的 特征集更新
    self.metric.partial_fit(np.asarray(features), np.asarray(targets),
                            active_targets)
```

进行检测结果和跟踪预测结果的匹配(级联匹配)

1. 将已存在的tracker分为confirmed tracks和unconfirmed tracks
1. 针对之前已经confirmed tracks，将它们与当前的检测结果进行级联匹配(这个匹配操作需要从刚刚匹配成功的tracker循环遍历到最多已经有30次没有匹配的tracker，这样做是为了对更加频繁出现的目标赋予优先权)
1. unconfirmed tracks和unmatched tracks一起组成iou_track_candidates，与还没有匹配的检测结果unmatched_detections进行IOU匹配

match函数
```python
# tracker.py
# 级联匹配
def _match(self, detections):
    # 主要功能是进行匹配，找到匹配的，未匹配的部分
    def gated_metric(tracks, dets, track_indices, detection_indices):
        # 功能： 用于计算track和detection之间的距离，代价函数
        #        需要使用在KM算法之前
        # 调用：
        # cost_matrix = distance_metric(tracks, detections,
        #                  track_indices, detection_indices)
        features = np.array([dets[i].feature for i in detection_indices])
        targets = np.array([tracks[i].track_id for i in track_indices])

        # 1. 通过最近邻计算出代价矩阵 cosine distance
        cost_matrix = self.metric.distance(features, targets)
        # 2. 计算马氏距离,得到新的状态矩阵
        cost_matrix = linear_assignment.gate_cost_matrix(
            self.kf, cost_matrix, tracks, dets, track_indices,
            detection_indices)
        return cost_matrix

    # Split track set into confirmed and unconfirmed tracks.
    # 划分不同轨迹的状态
    confirmed_tracks = [
        i for i, t in enumerate(self.tracks) if t.is_confirmed()
    ]
    unconfirmed_tracks = [
        i for i, t in enumerate(self.tracks) if not t.is_confirmed()
    ]

    # 进行级联匹配，得到匹配的track、不匹配的track、不匹配的detection
    '''
    !!!!!!!!!!!
    级联匹配
    !!!!!!!!!!!
    '''
    # gated_metric->cosine distance
    # 仅仅对确定态的轨迹进行级联匹配
    matches_a, unmatched_tracks_a, unmatched_detections = \
        linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks)

    # 将所有状态为未确定态的轨迹和刚刚没有匹配上的轨迹组合为iou_track_candidates，
    # 进行IoU的匹配
    iou_track_candidates = unconfirmed_tracks + [
        k for k in unmatched_tracks_a
        if self.tracks[k].time_since_update == 1  # 刚刚没有匹配上
    ]
    # 未匹配
    unmatched_tracks_a = [
        k for k in unmatched_tracks_a
        if self.tracks[k].time_since_update != 1  # 已经很久没有匹配上
    ]

    '''
    !!!!!!!!!!!
    IOU 匹配
    对级联匹配中还没有匹配成功的目标再进行IoU匹配
    !!!!!!!!!!!
    '''
    # 虽然和级联匹配中使用的都是min_cost_matching作为核心，
    # 这里使用的metric是iou cost和以上不同
    matches_b, unmatched_tracks_b, unmatched_detections = \
        linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections)

    matches = matches_a + matches_b  # 组合两部分match得到的结果

    unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
    return matches, unmatched_tracks, unmatched_detections
```

级联匹配对应的实现
```python
# 1. 分配track_indices和detection_indices
if track_indices is None:
    track_indices = list(range(len(tracks)))

if detection_indices is None:
    detection_indices = list(range(len(detections)))

unmatched_detections = detection_indices

matches = []
# cascade depth = max age 默认为70
for level in range(cascade_depth):
    if len(unmatched_detections) == 0:  # No detections left
        break

    track_indices_l = [
        k for k in track_indices
        if tracks[k].time_since_update == 1 + level
    ]
    if len(track_indices_l) == 0:  # Nothing to match at this level
        continue

    # 2. 级联匹配核心内容就是这个函数
    matches_l, _, unmatched_detections = \
        min_cost_matching(  # max_distance=0.2
            distance_metric, max_distance, tracks, detections,
            track_indices_l, unmatched_detections)
    matches += matches_l
unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
```
**门控矩阵**
门控矩阵的作用就是通过计算卡尔曼滤波的状态分布和测量值之间的距离对代价矩阵进行限制。

代价矩阵中的距离是Track和Detection之间的表观相似度，假如一个轨迹要去匹配两个表观特征非常相似的Detection，这样就很容易出错，但是这个时候分别让两个Detection计算与这个轨迹的马氏距离，并使用一个阈值gating_threshold进行限制，所以就可以将马氏距离较远的那个Detection区分开，可以降低错误的匹配。
```python
def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
    # 根据通过卡尔曼滤波获得的状态分布，使成本矩阵中的不可行条目无效。
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]  # 9.4877

    measurements = np.asarray([detections[i].to_xyah()
                               for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance >
                    gating_threshold] = gated_cost  # 设置为inf
    return cost_matrix
```

**IOU匹配**(An intersection over union distance metric)<br>
计算跟踪框两两之间的IOU，输出格式为ndarray的代价矩阵
程序中把cost大于阈值(0.7)的，都置成了0.7
```python
# iou_matching.py
def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix
```

**匈牙利算法**(Hungarian Algorithm)
输出格式为(List\[(int, int)\], List\[int\], List\[int\])
1. 匹配的跟踪框(track)和检测框(detection)列表索引
1. 不匹配的跟踪框(track)列表索引
1. 不匹配的检测框(detection)列表索引
```python
# linear_assignment.py
def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    
    # 把cost_matrix作为匈牙利算法的输入，得到线性匹配结果
    indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        
        # 如果某个组合的cost大于阈值，这样的组合仍然unmatched
        # 需要将组合中的检测框和跟踪框放回各自的unmatched列表
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    # 经过上述处理后，得到依据IOU的当然匹配结果
    return matches, unmatched_tracks, unmatched_detections
```
#### 卡尔曼滤波更新阶段
对于每个匹配成功的track，用其对应的detection进行更新，并处理未匹配tracks和detections<br>
根据匹配情况进行后续相应操作

更新的公式
```python
def project(self, mean, covariance):
    # R 测量过程中噪声的协方差
    std = [
        self._std_weight_position * mean[3],
        self._std_weight_position * mean[3],
        1e-1,
        self._std_weight_position * mean[3]]

    # 初始化噪声矩阵R
    innovation_cov = np.diag(np.square(std))

    # 将均值向量映射到检测空间，即Hx'
    mean = np.dot(self._update_mat, mean)

    # 将协方差矩阵映射到检测空间，即HP'H^T
    covariance = np.linalg.multi_dot((
        self._update_mat, covariance, self._update_mat.T))

    return mean, covariance + innovation_cov

def update(self, mean, covariance, measurement):
    # 通过估计值和观测值估计最新结果

    # 将均值和协方差映射到检测空间，得到 Hx' 和 S
    projected_mean, projected_cov = self.project(mean, covariance)

    # 矩阵分解
    chol_factor, lower = scipy.linalg.cho_factor(
        projected_cov, lower=True, check_finite=False)

    # 计算卡尔曼增益K
    kalman_gain = scipy.linalg.cho_solve(
        (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
        check_finite=False).T

    # z - Hx'
    innovation = measurement - projected_mean

    # x = x' + Ky
    new_mean = mean + np.dot(innovation, kalman_gain.T)

    # P = (I - KH)P'
    new_covariance = covariance - np.linalg.multi_dot((
        kalman_gain, projected_cov, kalman_gain.T))
    return new_mean, new_covariance
```
这个公式中，z是Detection的mean，不包含变化值，状态为[cx,cy,a,h]。H是测量矩阵，将Track的均值向量映射到检测空间。计算的y是Detection和Track的均值误差。

R是目标检测器的噪声矩阵，是一个4x4的对角矩阵。对角线上的值分别为中心点两个坐标以及宽高的噪声。

计算的是卡尔曼增益，是作用于衡量估计误差的权重。

更新后的均值向量x。

更新后的协方差矩阵。

1. 对于matched组合，要用检测结果去更新相应tracker的参数
```python
# tracker.py
 for track_idx, detection_idx in matches:
    self.tracks[track_idx].update(self.kf, detections[detection_idx])
```

更新包括以下三个操作
    1. 更新卡尔曼滤波的一系列运动变量，命中次数以及重置时间
    1. 将检测框的深度特征保存到此跟踪框的特征集合中
    1. 如果已经连续命中3帧，将跟踪框的状态由tentative改为confirmed
```python
# track.py
def update(self, kf, detection):
    self.mean, self.covariance = kf.update(
        self.mean, self.covariance, detection.to_xyah())
    self.features.append(detection.feature)
    self.hits += 1
    self.time_since_update = 0
    if self.state == TrackState.Tentative and self.hits >= self._n_init:
        self.state = TrackState.Confirmed
```
2. 对于不匹配的跟踪框
```python
# tracker.py
for track_idx in unmatched_tracks:
    self.tracks[track_idx].mark_missed()
```
有以下两种情况
    1. 如果这个跟踪框是还未确认的，直接将其从跟踪列表删除
    1. 如果这个跟踪框是之前经过确认的，但是已经连续max_age(3)帧没能匹配到检测结果了，也需要将其从跟踪列表删除

```python
# track.py
def mark_missed(self):
    if self.state == TrackState.Tentative:
        self.state = TrackState.Deleted
    elif self.time_since_update > self._max_age:
        self.state = TrackState.Deleted
```

3. 对于不匹配的检测框，要为其创建新的追踪器(tracker)
```python
# tracker.py
for detection_idx in unmatched_detections:
    self._initiate_track(detections[detection_idx])
```
根据初始检测位置初始化新的卡尔曼滤波器的平均值(mean)和协方差(covariance)
```python
# tracker.py
def _initiate_track(self, detection):
    mean, covariance = self.kf.initiate(detection.to_xyah())
    # 初始化一个新的tracker
    self.tracks.append(Track(
        mean, covariance, self._next_id, self.n_init, self.max_age,
        detection.feature))
    self._next_id += 1
```
track初始化
```python
# track.py
def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
    # tracker的构造函数
    self.mean = mean # 初始的mean
    self.covariance = covariance # 初始的covariance
    self.track_id = track_id
    self.hits = 1
    self.age = 1
    self.time_since_update = 0 # 初始值为0

    self.state = TrackState.Tentative # 初始为Tentative状态
    self.features = []
    if feature is not None:
        self.features.append(feature) # 相应的det特征存入特征库中 

    self._n_init = n_init
    self._max_age = max_age
```
最后需要删除待删除状态的追踪器
```python
# tracker.py
self.tracks = [t for t in self.tracks if not t.is_deleted()]
```
### 更新已经确认的追踪器特征集并输出已经确认的追踪器的跟踪预测结果
tracker最多保存最近与之匹配的100帧检测结果的特征集
```python
# tracker.py
active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
features, targets = [], []
for track in self.tracks:
    if not track.is_confirmed():
        continue
    features += track.features
    targets += [track.track_id for _ in track.features]
    track.features = []
self.metric.partial_fit(
    np.asarray(features), np.asarray(targets), active_targets)
```

```python
# nn_matching.py
def partial_fit(self, features, targets, active_targets):
    # 每个activate的追踪器保留最近的self.budget条特征
    for feature, target in zip(features, targets):
        self.samples.setdefault(target, []).append(feature)
        if self.budget is not None:
            self.samples[target] = self.samples[target][-self.budget:]
    # 以dict的形式插入总库
    self.samples = {k: self.samples[k] for k in active_targets}
```

### 类图
![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/object%20detection/demo/iiffneust4.png)

DeepSort是核心类，调用其他模块，大体上可以分为三个模块：

* ReID模块，用于提取表观特征，原论文中是生成了128维的embedding。
* Track模块，轨迹类，用于保存一个Track的状态信息，是一个基本单位。
* Tracker模块，Tracker模块掌握最核心的算法，卡尔曼滤波和匈牙利算法都是通过调用这个模块来完成的。

### 运行结果
采用Faster R-CNN网络进行目标检测，检测的置信度阈值设置为0.3，λ=0,Amax=30。<br>
在每一帧中已经确认的tracker结果将会标注在图上，下图所示为MOT16-06中某一帧已确认的结果：<br>
![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/deepsort/images/Figure%20MOT16-06.png)<br>
所有MOT16的追踪处理结果已生成为文本文档(https://github.com/barryyan0121/MOT_Comparison/tree/master/deepsort/results)<br>
其格式为标准MOTChallenge检测格式，第一项为当前帧数，第二项为目标ID，第三至第六项为检测框特征\[center x, center y, aspect ratio, height\]，第七至第十项均为-1。

MOT16-01共计450帧数，耗时22.47秒，平均每秒处理20.0帧，生成视频为29FPS

MOT16-03共计1500帧数，耗时138.76秒，平均每秒处理10.8帧，生成视频为29FPS

MOT16-06共计1194帧数，耗时37.49秒，平均每秒处理31.8帧，生成视频为14FPS

MOT16-07共计500帧数，耗时33.37秒，平均每秒处理15.0帧，生成视频为29FPS

MOT16-08共计625帧数，耗时32.99秒，平均每秒处理18.9帧，生成视频为29FPS

MOT16-12共计900帧数，耗时36.21秒，平均每秒处理24.9帧，生成视频为29FPS

MOT16-14共计750帧数，耗时43.96秒，平均每秒处理17.1帧，生成视频为10FPS

除MOT16-14外，其余生成视频皆接近于原FPS

### 改进策略

第一点，把Re-ID网络和检测网络融合，做一个精度和速度的trade off。

第二点，对于轨迹段来说，时间越长的轨迹是不是更应该得到更多的信任，不仅仅只是级联匹配的优先级，由此可以引入轨迹评分的机制。

第三点，从直觉上来说，检测和追踪是两个相辅相成的问题，良好的追踪可以弥补检测的漏检，良好的检测可以防止追踪的轨道飘逸，用预测来弥补漏检这个问题在DeepSORT里也并没有考虑。

第四点，DeepSORT里给马氏距离也就是运动模型设置的系数为0，也就是说在相机运动的情况下线性速度模型并不准确，所以可能可以找到更好的运动模型。

## FairMOT (A simple baseline for one-shot Multi-Object Tracking)

### 导语
FairMOT是类似于CenterTrack的基于CenterNet的联合检测和跟踪的框架，同时类似却又不同于JDE的框架，它探讨了检测框架与ReID特征任务的集成问题，这类框架被称为为one-shot MOT框架。

anchor-based的检测框架中存在anchor和特征的不对齐问题，所以这方面不如anchor-free框架，因而选择anchor-free算法——CenterNet，不过其用法并不是类似于CenterTrack，而是采用的Tracktor++的方式。

### Tracker++
Tracktor++算法是去年出现的一类全新的联合检测和跟踪的框架，这类框架与MOTDT框架最大的不同在于，检测部分不仅仅用于前景和背景的进一步分类，还利用回归对目标进行了进一步修正，其核心在于利用跟踪框和观测框代替原有的RPN模块，从而得到真正的观测框，最后利用数据关联实现跟踪框和观测框的匹配。流程图如下：

![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/object%20detection/demo/v2-fb2ddc1ea3290991400cb76f424e8fc1_720w.jpg)

原始的anchor-free框架的大多数backbone都是采用了骨骼关键点中的hourglass结构：

## 目标检测(Object Detection)
如何从图像中解析出可供计算机理解的信息，是机器视觉的中心问题。近年来，深度学习模型逐渐取代传统机器视觉方法而成为目标检测领域的主流算法。深度学习模型由于其强大的表示能力，加之数据量的积累和计算力的进步，成为机器视觉的热点研究方向。

那么，如何理解一张图片？根据后续任务的需要，有三个主要的层次。
![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/object%20detection/images/three_level.jpg)<br>
一是分类（Classification）。分类即是将图像结构化为某一类别的信息，用事先确定好的类别(string)或实例ID来描述图片。这一任务是最简单、最基础的图像理解任务，也是深度学习模型最先取得突破和实现大规模应用的任务。其中，ImageNet是最权威的评测集，每年的ILSVRC催生了大量的优秀深度网络结构，为其他任务提供了基础。在应用领域，人脸、场景的识别等都可以归为分类任务。

二是检测（Detection）。分类任务关心整体，给出的是整张图片的内容描述，而检测则关注特定的物体目标，要求同时获得这一目标的类别信息和位置信息。相比分类，检测给出的是对图片前景和背景的理解，我们需要从背景中分离出感兴趣的目标，并确定这一目标的描述（类别和位置），因而，检测模型的输出是一个列表，列表的每一项使用一个数据组给出检出目标的类别和位置（常用矩形检测框的坐标表示）。

三是分割（Segmentation）。分割包括语义分割（semantic segmentation）和实例分割（instance segmentation），前者是对前背景分离的拓展，要求分离开具有不同语义的图像部分，而后者是检测任务的拓展，要求描述出目标的轮廓（相比检测框更为精细）。分割是对图像的像素级描述，它赋予每个像素类别（实例）意义，适用于理解要求较高的场景，如无人驾驶中对道路和非道路的分割。

目标检测，即为图像理解的中层次。

![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/object%20detection/images/object_detection.jpg)

在经典深度学习模型中，目标检测分为单阶段检测模型(1-stage)和两阶段检测模型(2-stage)。两阶段模型因其对图片的两阶段处理得名，也称为基于区域(Region-based)的方法。单阶段检测模型没有中间的区域检出过程，直接从图片获得预测结果，也被成为Region-free方法。

### R-CNN/Fast R-CNN/Faster R-CNN
R-CNN系列模型为两阶段模型的代表。

传统的计算机视觉方法常用精心设计的手工特征(如SIFT, HOG)描述图像，而深度学习的方法则倡导习得特征，从图像分类任务的经验来看，CNN网络自动习得的特征取得的效果已经超出了手工设计的特征。本篇在局部区域应用卷积网络，以发挥卷积网络学习高质量特征的能力。

R-CNN网络结构

![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/object%20detection/images/r-cnn.jpg)

R-CNN将检测抽象为两个过程，一是基于图片提出若干可能包含物体的区域(即图片的局部裁剪，被称为Region Proposal)，文中使用的是Selective Search算法；二是在提出的这些区域上运行当时表现最好的分类网络(AlexNet)，得到每个区域内物体的类别。

输入CNN前，我们需要根据Ground Truth对提出的Region Proposal进行标记，这里使用的指标是IoU(Intersection over Union，交并比)。IoU计算了两个区域之交的面积跟它们之并的比，描述了两个区域的重合程度。<br>
![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/object%20detection/images/IOU.jpg)

R-CNN的想法直接明了，即将检测任务转化为区域上的分类任务，是深度学习方法在检测任务上的试水。模型本身存在的问题也很多，如需要训练三个不同的模型(proposal, classification, regression)、重复计算过多导致的性能问题等。尽管如此，这篇论文的很多做法仍然广泛地影响着检测任务上的深度模型革命，后续的很多工作也都是针对改进这一工作而展开。

Fast R-CNN(共享卷积网络)提出将基础网络在图片整体上运行完毕后，再传入R-CNN子网络，共享了大部分计算，故有Fast之名。<br>
![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/object%20detection/images/fast%20r-cnn.jpg)

上图是Fast R-CNN的架构。图片经过feature extractor得到feature map, 同时在原图上运行Selective Search算法并将RoI(Region of Interset，实为坐标组，可与Region Proposal混用)映射到到feature map上，再对每个RoI进行RoI Pooling操作便得到等长的feature vector，将这些得到的feature vector进行正负样本的整理(保持一定的正负样本比例)，分batch传入并行的R-CNN子网络，同时进行分类和回归，并将两者的损失统一起来。

Fast R-CNN的这一结构正是检测任务主流2-stage方法所采用的元结构的雏形。它将Proposal, Feature Extractor, Object Classification&Localization统一在一个整体的结构中，并通过共享卷积计算提高特征利用效率。

Faster R-CNN是2-stage方法的奠基性工作，提出的RPN网络取代Selective Search算法使得检测任务可以由神经网络端到端地完成。粗略的讲，Faster R-CNN = RPN + Fast R-CNN，跟RCNN共享卷积计算的特性使得RPN引入的计算量很小，使得Faster R-CNN可以在单个GPU上以5fps的速度运行，而在精度方面达到当前最佳(State of the Art)。

Faster R-CNN主要贡献是提出Regional Proposal Networks，替代之前的SS算法。RPN网络将Proposal这一任务建模为二分类（是否为物体）的问题。

![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/object%20detection/images/faster%20r-cnn.jpg)

第一步是在一个滑动窗口上生成不同大小和长宽比例的anchor box(如上图右边部分)，取定IoU的阈值，按Ground Truth标定这些anchor box的正负。于是，传入RPN网络的样本数据被整理为anchor box(坐标)和每个anchor box是否有物体(二分类标签)。RPN网络将每个样本映射为一个概率值和四个坐标值，概率值反应这个anchor box有物体的概率，四个坐标值用于回归定义物体的位置。最后将二分类和坐标回归的损失统一起来，作为RPN网络的目标训练。由RPN得到Region Proposal在根据概率值筛选后经过类似的标记过程，被传入R-CNN子网络，进行多分类和坐标回归，同样用多任务损失将二者的损失联合。

Faster R-CNN的成功之处在于用RPN网络完成了检测任务的"深度化"。使用滑动窗口生成anchor box的思想也在后来的工作中越来越多地被采用(YOLO v2等)。这项工作奠定了"RPN+RCNN"的两阶段方法元结构，影响了大部分后续工作。

### YOLO(You Only Look Once)
YOLO是单阶段方法的开山之作。它将检测任务表述成一个统一的、端到端的回归问题，并且以只处理一次图片同时得到位置和分类而得名。

#### YOLO的主要优点：
* 快
* 全局处理使得背景错误相对少，相比基于局部(区域)的方法，比如Fast RCNN
* 泛化性能好，在艺术作品上做检测时，YOLO表现比Fast R-CNN好

![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/object%20detection/images/yolo.jpg)

#### YOLO的工作流程如下：

1. 准备数据：将图片缩放，划分为等分的网格，每个网格按跟Ground Truth的IoU分配到所要预测的样本。

2. 卷积网络：由GoogLeNet更改而来，每个网格对每个类别预测一个条件概率值，并在网格基础上生成B个box，每个box预测五个回归值，四个表征位置，第五个表征这个box含有物体（注意不是某一类物体）的概率和位置的准确程度（由IoU表示）。测试时，分数如下计算：

![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/object%20detection/images/convnet.jpg)

等式左边第一项由网格预测，后两项由每个box预测，以条件概率的方式得到每个box含有不同类别物体的分数。 因而，卷积网络共输出的预测值个数为S×S×(B×5+C)，其中S为网格数，B为每个网格生成box个数，C为类别数。

3. 后处理：使用NMS（Non-Maximum Suppression，非极大抑制）过滤得到最后的预测框

#### 损失函数的设计
![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/object%20detection/images/loss_function.jpg)

损失函数被分为三部分：坐标误差、物体误差、类别误差。为了平衡类别不均衡和大小物体等带来的影响，损失函数中添加了权重并将长宽取根号。

YOLO提出了单阶段的新思路，相比两阶段方法，其速度优势明显，实时的特性令人印象深刻。但YOLO本身也存在一些问题，如划分网格较为粗糙，每个网格生成的box个数等限制了对小尺度物体和相近物体的检测。

### EfficientNet/EfficientDet
EfficientDet是Google的大作。在分类任务上有一篇EfficientNet，从名字看就知道，它是EfficientNet的在目标检测任务上的延伸。这篇文章的重点有两个：首先是BiFPN结构(weighted bi-directional feature pyramid network)，可以更快更好地融合特征。其次是提出一种compound scaling method，在EfficientNet那篇论文里也有提过。本质上，就是把NAS需要搜索优化的很多参数，基于一些insight和经验，用少量的参数关联起来，这样就可以减小减小搜索空间，实现更快更高效地搜索。EfficientDet使用的是SSD+FPN的one-stage检测架构，所以需要搜索的网络结构参数，包含backbone、feature网络(FPN)、bbox/cls 网络的width、height以及输入的resolution。

![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/object%20detection/images/efficientdet.jpg)

#### BiFPN (双向FPN)

![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/object%20detection/images/bifpn.jpg)

FPN只有bottom-2-up的path；PANet使用了双path的结构；NAS-FPN通过神经架构搜索得到网络结构，但是结构的可解释性很差。EfficientDet参考PANet，增加了skip connection和weighted fusion，以便更好地融合特征。

#### Weighted Feature Fusion
![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/object%20detection/images/weighted%20feature%20fusion.jpg)

在FPN部分，每个节点都是由多个节点融合而来的，我们发现，不同深度的feature map对结果的贡献是不同的。因此，我们给每个节点的输入节点添加learnable权重。为了更好地学习降低计算效率，不适用sigmoid归一化，而使用均值归一化。

#### Compound Scaling Method

目标检测中需要考虑的参数比分类任务更多。EfficientNet分类任务中只考虑了网络三要素width，depth和resolution(input)，目标检测任务中，还需要考虑cls/bbox net。EfficientDet将EfficientNet拿来做backbone，从而有效控制其规模，neck部分，BiFPN的channel数量、重复的layer数量也可以控制，此外还有head部分的层数，以及输入图片的分辨率(input resolution)，这些组成了EfficientDet的Compound Scaling。

通过优化一个参数关联所有需要搜索优化的参数搜索得到最优的网络架构，这就是compound scaling method。

#### 网络结构
![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/object%20detection/images/efficientdet%20architecture.jpg)

基于一阶段SSD+FPN结构改造。以EfficientNet为backbone，然后接上3个(bottom-up & up-down)的结构，最后的特征用于预测bbox和cls。

* Backbone network：直接利用EfficientNet的B0-B6作为预训练的backbone
* BiFPN network：指数调整BiFPN的channel数，线性调整BiFPN的depth
* Box/class prediction network：channel数和BiFPN保持一致，线性调整depth
* Input image resolution：因为使用了P3-P7层进行特征融合，输入分辨率调整后必须是128的倍数

EfficientDet的调整策略总结如下：<br>
![Image of pic](https://github.com/barryyan0121/MOT_Comparison/blob/master/object%20detection/images/adjustment.jpg)

一系列的EfficientDet网络都在精度、参数量、计算量、CPU速度以及GPU速度上完成了对之前SOTA方法的提升。在相同精度要求下，EfficientDet比YOLOv3少28倍的计算量，比RetinaNet少30倍的计算量，比Nas-FPN少19倍的计算量。此外，在刷SOTA结果时，单模型单尺度下EfficientDet-D7可以达到51.0 mAP，这比目前最好的结果还要高，同时参数量少了4倍，计算量少了9.3倍。

## 行人重识别(Re-ID)

## 参考资源
https://zhuanlan.zhihu.com/p/59148865<br>
https://zhuanlan.zhihu.com/p/90835266<br>
https://zhuanlan.zhihu.com/p/80764724<br>
https://zhuanlan.zhihu.com/p/114349651<br>
https://www.cnblogs.com/yanwei-li/p/8643446.html<br>
https://blog.csdn.net/cdknight_happy/article/details/79731981<br>
https://zhuanlan.zhihu.com/p/34142321<br>
https://zhuanlan.zhihu.com/p/131008921<br>
https://zhuanlan.zhihu.com/p/31921944<br>
https://zhuanlan.zhihu.com/p/126558285<br>
https://cloud.tencent.com/developer/article/1618058
