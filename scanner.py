#-*- coding: utf-8 -*-
"""
scanner.py

"""

# ============  package import ============
import numpy as np
import cv2, sys, time

# ============  basic set============
# 人脸区域检测模型配置
face_cascade = cv2.CascadeClassifier('./3dpart/opencv/haarcascade_frontalface_default.xml')

# ============  function ============
class SliderWindow():
    """使用滑动窗口遍历图像的每个小区块"""
    @staticmethod
    def run(cv2_image, stepSizeX, stepSizeY, windowSizeW, windowSizeH):
        """
            cv2_image: opencv mat 图像矩阵
            stepSizeX:
            stepSizeY:
            windowSize: (w, h)
        """
        # slide a window across the image
        for y in xrange(0, cv2_image.shape[0], stepSizeY):
            for x in xrange(0, cv2_image.shape[1], stepSizeX):
                # yield the current window
                yield (x, y, cv2_image[y:y + windowSizeH, x:x + windowSizeW])

class FaceScanner():
    """从图像中定位可能的人脸区域"""
    @staticmethod
    def detect(cv2_image):
        """
            cv2_image: opencv mat 图像矩阵
            face_regions: list of coordinates, 每个坐标是一个tuple, (x, y, w, h) 分别表示x,y坐标和宽高.
        """
        # 灰度化
        gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        # 级联算法获取人脸位置
        face_regions = face_cascade.detectMultiScale(gray, 1.3, 5)
        return face_regions


# ============  function test ============
def test_sliding_window(file_path):
    image = cv2.imread(file_path)
    winW, winH, stepSizeX, stepSizeY = 60, 60, 40, 50
    for (x, y, window) in sliding_window(image, stepSizeX=stepSizeX, stepSizeY=stepSizeY, windowSizeW=winW, windowSizeH=winH):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        copy_image = image.copy()
        cv2.rectangle(copy_image, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", copy_image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
