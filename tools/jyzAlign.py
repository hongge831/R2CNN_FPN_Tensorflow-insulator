import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

src = 'D:/jyzdata/jyzbig/'
fileList = os.listdir(src)
jsonList = [x for x in fileList if x.endswith('json')]
# temp = jsonList[2]


def align(num,filename):
    src = 'D:/jyzdata/jyzbig/'
    desrc = 'D:/jyzdata/jyzAlign/'
    with open(src + filename, 'r') as f:
        #读取json文件
        jsonfile = json.loads(f.read())
        #读取json文件中对应的图片信息
        img = cv2.imread(src + jsonfile['imagePath'])
        for idx in range(len(jsonfile['shapes'])):
            #根据图片中的关键点坐标计算最小的包络矩形
            cnt = np.asarray(jsonfile['shapes'][idx]['points'])
            cnt = cnt.astype(int)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x 获取最小外接矩形的4个顶点
            box = np.int0(box)
            # print(str(idx) + src + jsonfile['imagePath'])
            # cv2.circle(img, (box[0][0], box[0][1]), 50, (255, 0, 0), 10)  # 修改最后一个参数
            # cv2.circle(img, (box[1][0], box[1][1]), 50, (0, 255, 0), 10)  # 修改最后一个参数
            # cv2.circle(img, (box[2][0], box[2][1]), 50, (0, 0, 255), 10)  # 修改最后一个参数
            # cv2.circle(img, (box[3][0], box[3][1]), 50, (0, 0, 0), 10)  # 修改最后一个参数
            #这里主要针对不同的矩形倾斜角度做分类，使得几乎所有的情况都能旋转至水平
            #rect[1][0]是矩形所对应的宽，rect[1][1]对应的是矩形的长
            if (rect[1][0]<rect[1][1]):
                RotateMatrix = cv2.getRotationMatrix2D(center=(box[0][0], box[0][1]), angle=(90 + rect[2]), scale=1)
                if(abs(rect[2])>50):
                    addition = rect[1][1]//3
                else:
                    addition = rect[1][1]
                RotateMatrix[0, 2] += addition
                # RotateMatrix[1, 2] += rect[1][1]
                RotImg = cv2.warpAffine(img, RotateMatrix, (img.shape[0] * 2, img.shape[1]))  # 为了不让尺寸的图像部分消失
                y1 = box[0][1]
                y0 = y1 - rect[1][0]
                x0 = box[0][0]+addition
                x1 = x0 - rect[1][1]
                #裁剪旋转后的图像
                cropImg = RotImg[int(y0):int(y1), int(x1):int(x0)]
                #将裁剪后的图像保存至本地
                cv2.imwrite(desrc + str(num) +str(idx) + '_' + jsonfile['shapes'][idx]['label'] + '.jpg', cropImg)
            else:
                RotateMatrix = cv2.getRotationMatrix2D(center=(box[0][0], box[0][1]), angle=rect[2], scale=1)
                RotImg = cv2.warpAffine(img, RotateMatrix, (img.shape[0] * 2, img.shape[1]))  # 为了不让尺寸的图像部分消失
                y0 = box[0][1] - rect[1][1]
                y1 = box[0][1]
                x0 = box[0][0]
                x1 = box[0][0] + rect[1][0]
                cropImg = RotImg[int(y0):int(y1), int(x0):int(x1)]
                cv2.imwrite(desrc + str(num) +str(idx) + '_' + jsonfile['shapes'][idx]['label'] + '.jpg', cropImg)

if __name__ == '__main__':
    for i, filename in tqdm(enumerate(jsonList)):
        align(i,filename)
    # align(100, '1520921045284-jueyuanzi.json')

