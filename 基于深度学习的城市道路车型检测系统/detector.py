import cv2
import torch
import numpy as np
import os

class YOLOv8CarDetector:
    """
    只加载用户提供的 best.pt，不额外下载 yolo.pt
    """
    def __init__(self, weight_path='best.pt', conf_thres=0.4, iou_thres=0.45):
        self.conf_thres = conf_thres
        self.iou_thres  = iou_thres

        # 1. 加载本地模型
        assert os.path.exists(weight_path), f'{weight_path} 不存在！'
        self.model = torch.hub.load('ultralytics/yolov8', 'custom',
                                    path=weight_path, force_reload=False)
        self.model.conf = conf_thres
        self.model.iou  = iou_thres

        # 2. 获取类别名
        self.names = self.model.names if hasattr(self.model, 'names') else [
            'car', 'bus', 'truck']   # 按需改成自己的类别

    def detect(self, img_bgr):
        """
        img_bgr: opencv 读取的 BGR 图
        返回:  list[ [x1,y1,x2,y2,conf,cls_id], ... ]
        """
        results = self.model(img_bgr)
        det = results.xyxy[0].cpu().numpy()   # [[x1,y1,x2,y2,conf,cls], ...]
        return det

    def draw(self, img_bgr, dets, thickness=2, font_scale=0.6):
        """
        把框和类别画到图上
        """
        for *xyxy, conf, cls in dets:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f'{self.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            cv2.putText(img_bgr, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness=2)
        return img_bgr