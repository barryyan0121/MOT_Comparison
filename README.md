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
按视频帧顺序处理，每一帧的处理流程如下:
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
    # 矩阵Q(系统误差)
    motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
    # 卡尔曼滤波公式1
    mean = np.dot(self._motion_mat, mean)
    
    # 卡尔曼滤波公式2
    covariance = np.linalg.multi_dot((
        self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

    return mean, covariance
```

mean : 格式为ndarray的位于前一个时间点的目标状态八维向量   
covariance : 格式为ndarray的位于前一个时间点的目标状态8x8的协方差矩阵
输出格式为(ndarray, ndarray)的预测目标平均向量和协方差矩阵，未被观测的速度将被初始化为0

#### 匹配
首先对基于外观信息的马氏距离(Mahalanobis distance)计算跟踪框(tracks)和检测框(detections)的代价矩阵，然后相继进行**级联匹配**和**IOU匹配**，最后得到当前帧的所有匹配对、未匹配的tracks以及未匹配的detections。

```python
# deep_sort_app.py
tracker.update(detections)
```

```python
# tracker.py
def update(self, detections):
      
    # 运行级联匹配
    matches, unmatched_tracks, unmatched_detections = self._match(detections)

    # 更新跟踪框代价矩阵
    for track_idx, detection_idx in matches:
        self.tracks[track_idx].update(
            self.kf, detections[detection_idx])
    for track_idx in unmatched_tracks:
        self.tracks[track_idx].mark_missed()
    for detection_idx in unmatched_detections:
        self._initiate_track(detections[detection_idx])
    self.tracks = [t for t in self.tracks if not t.is_deleted()]

    # 更新检测框代价矩阵
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

进行检测结果和跟踪预测结果的匹配(级联匹配)

1. 将已存在的tracker分为confirmed tracks和unconfirmed tracks
1. 针对之前已经confirmed tracks，将它们与当前的检测结果进行级联匹配(这个匹配操作需要从刚刚匹配成功的tracker循环遍历到最多已经有30次没有匹配的tracker，这样做是为了对更加频繁出现的目标赋予优先权)
1. unconfirmed tracks和unmatched tracks一起组成iou_track_candidates，与还没有匹配的检测结果unmatched_detections进行IOU匹配
```python
# tracker.py
# 级联匹配
def _match(self, detections):

    def gated_metric(tracks, dets, track_indices, detection_indices):
        features = np.array([dets[i].feature for i in detection_indices])
        targets = np.array([tracks[i].track_id for i in track_indices])
        cost_matrix = self.metric.distance(features, targets)
        cost_matrix = linear_assignment.gate_cost_matrix(
            self.kf, cost_matrix, tracks, dets, track_indices,
            detection_indices)

        return cost_matrix

    # 将已存在的tracker分为confirmed tracks和unconfirmed tracks
    confirmed_tracks = [
        i for i, t in enumerate(self.tracks) if t.is_confirmed()]
    unconfirmed_tracks = [
        i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

    # 针对之前已经confirmed tracks，将它们与当前的检测结果进行级联匹配
    matches_a, unmatched_tracks_a, unmatched_detections = \
        linear_assignment.matching_cascade(
            gated_metric, self.metric.matching_threshold, self.max_age,
            self.tracks, detections, confirmed_tracks)

    # unconfirmed tracks和unmatched tracks一起组成iou_track_candidates
    # 与还没有匹配的检测结果unmatched_detections进行IOU匹配
    iou_track_candidates = unconfirmed_tracks + [
        k for k in unmatched_tracks_a if
        self.tracks[k].time_since_update == 1]
    unmatched_tracks_a = [
        k for k in unmatched_tracks_a if
        self.tracks[k].time_since_update != 1]
    matches_b, unmatched_tracks_b, unmatched_detections = \
        linear_assignment.min_cost_matching(
            iou_matching.iou_cost, self.max_iou_distance, self.tracks,
            detections, iou_track_candidates, unmatched_detections)

    matches = matches_a + matches_b
    unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
    return matches, unmatched_tracks, unmatched_detections
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

### 运行结果
生成视频帧数 处理速度 结果为文本文档

### 改进策略

第一点，把Re-ID网络和检测网络融合，做一个精度和速度的trade off。

第二点，对于轨迹段来说，时间越长的轨迹是不是更应该得到更多的信任，不仅仅只是级联匹配的优先级，由此可以引入轨迹评分的机制。

第三点，从直觉上来说，检测和追踪是两个相辅相成的问题，良好的追踪可以弥补检测的漏检，良好的检测可以防止追踪的轨道飘逸，用预测来弥补漏检这个问题在DeepSORT里也并没有考虑。

第四点，DeepSORT里给马氏距离也就是运动模型设置的系数为0，也就是说在相机运动的情况下线性速度模型并不准确，所以可能可以找到更好的运动模型。

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
