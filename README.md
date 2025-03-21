
## 基于Yolov8目标分类Deepsort目标追踪

代码采用C++实现，目标检测支持YOLOv8,跟踪支持deepsort 。
检测模型可以直接从YOLOv8官网，导出onnx使用
特征提取可以自己训练，导出onnx使用，onnxruntime cpu 推理，方便使用.
特征支持自定义维度例如 128,256,512等


#### 操作系统：Ubuntu 18.04
#### YOLV ：https://github.com/ultralytics/yolov8
#### 硬件：NVIDIA GeForce RTX 4060 Ti 8G
#### CUDA 12.6
#### CUDNN 9.6
#### onnxruntime 1.12.1


#### 2 模型配置

yolov8n.onnx 

coco_80_labels_list.txt
coco.yaml

#### 3 主函数
选择打开视频文件或者视频流等

```c
cv::VideoCapture capture("./1.mp4");
```

### 扩展方式
1 整体分为两部分，新增检测模块放置detector文件夹，新增跟踪模块放置tracker文件夹

## deepsort v1.0
### MOT using deepsort yolo3 with C++
操作系统：Windows11
编译环境：Cmake  3.30.0-rc4
硬件：INVIDIA Geforce RTC 3060
CUDA:12.6
Opencv:4.10.0
深度学习的模型分两块，一个是目标检测，另一个是目标跟踪
#### 目标检测的模型
地址：Yolov8 https://github.com/ultralytics/ultralytics
![File](images/file.png)
#### 目标跟踪模型
DeepSORT算法的主要步骤
![Track](images/track.png)
目标检测：DeepSORT算法依赖于目标检测器来确定视频中每一帧的目标位置。常用的目标检测器包括YOLO、Faster R-CNN等。检测器的输出通常包括目标的边界框（bounding box）和类别。

特征提取：DeepSORT使用深度学习模型来提取目标的外观特征。这些特征对于目标的再识别（re-identification，简称Re-ID）至关重要，因为即使目标在视频中被临时遮挡或丢失，这些特征也能帮助算法重新识别和关联目标。

匹配和跟踪：DeepSORT算法中的匹配过程涉及到计算检测框和预测框之间的相似度，并使用匈牙利算法来找到最优匹配。这个过程还包括卡尔曼滤波器的使用，它根据目标的历史运动信息来预测其在下一帧中的位置。

卡尔曼滤波：用于预测目标在下一帧中的位置。

匈牙利算法：用于计算检测框和预测框之间的最优匹配。

级联匹配：DeepSORT中的级联匹配是一种特殊的机制，它首先尝试将检测结果与高置信度的轨迹进行匹配，然后再与低置信度的轨迹进行匹配。这有助于提高匹配的准确性，尤其是在目标被遮挡或短暂消失时。

轨迹管理：DeepSORT维护每个目标的轨迹，并对新检测到的目标初始化新的轨迹。它还设置了确认状态（confirmed）和未确认状态（unconfirmed），以处理遮挡和临时丢失的情况。

### 物体速度计算
道路上的两条平行分界线上的四个点
//坐标转换
```C++
cv::Point2f src_points[] = {
           cv::Point2f(0,  359),//A
           cv::Point2f(639,  359),//B
           cv::Point2f(200,  160),//C
           cv::Point2f(400,  160) };//D
cv::Point2f dst_points[] = {
            cv::Point2f(0,  359),//A
            cv::Point2f(639,  359),//B
            cv::Point2f(0,  70),//C'
            cv::Point2f(639, 70) };//D'
Mat g_transform;
...
//计算放缩比例
g_transform = cv::getPerspectiveTransform(src_points, dst_points);
//根据比例，计算目标坐标用于计算距离和速度
cv::perspectiveTransform(points, transformed_points, g_transform);
```
![标定2](images/标定2.png)


## 记录一下后面要做的事情
研究一下模型的拓展，如何让模型有空间感知，能够根据驾驶的需求，
设计驾驶的路线，实现车辆的控制。


