from __future__ import division # 改变 Python 2 中除法操作符 / 的默认行为，使其表现得像 Python 3 中的除法操作符,结果会保留小数部分
import  matplotlib.pyplot as plt # 用于创建图表和可视化数据的 Python 库
import cv2
import os, glob # glob文件名匹配的模块
import numpy as np
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from Parking import Parking
import pickle   # 序列化和反序列化对象的标准模块

cwd = os.getcwd() # 获取当前工作目录

def img_process(test_images, park):
    # 过滤背景，低于lower_red和高于upper_red的部分分别编程0，lower_red~upper_red之间的值编程255
    # map 函数用于将一个函数应用到可迭代对象的每个元素，并返回结果
    # 通过 list 函数将其转换为列表
    white_yellow_images = list(map(park.select_rgb_white_yellow,test_images))
    park.show_images(white_yellow_images)

    # 转灰度图
    gray_images = list(map(park.convert_gray_scale, white_yellow_images))
    park.show_images(gray_images)

    # 进行边缘检测
    edge_images = list(map(lambda image: park.detect_edges(image),gray_images))
    park.show_images(edge_images)

    # 根据需要设定屏蔽区域
    roi_images = list(map(park.select_region, edge_images))
    park.show_images(roi_images)

    # 霍夫变换，得出直线
    list_of_lines= list(map(park.hough_line, roi_images))

    # zip 函数来同时迭代 test_images 和 list_of_lines 中的元素
    line_images = []
    for image,lines in zip(test_images,list_of_lines):
        line_images.append(park.draw_lines(image,lines))
    park.show_images(line_images)

    rect_images = []
    rect_coords = [] # 列矩形
    for image,lines in zip(test_images, list_of_lines):
         # 过滤部分直线，对直线进行排序，得出每一列的起始点和终止点，并将列矩形画出来
        new_image,rects = park.identify_blocks(image,lines)
        rect_images.append(new_image)
        rect_coords.append(rects)

    park.show_images(rect_images)

    delineated = []
    spot_pos = []
    for image,rects in zip(test_images, rect_coords):
        # 在图上将停车位画出来，并返回字典{坐标：车位序号}
        new_image,spot_dict = park.draw_parking(image,rects)
        delineated.append(new_image)
        spot_pos.append(spot_dict)

    park.show_images(delineated)
    final_spot_dict = spot_pos[1]

    print(len(final_spot_dict))

    with open('spot_dict.pickle','wb') as handle:
        pickle.dump(final_spot_dict,handle,property==pickle.HIGHEST_PROTOCOL)

    park.save_images_for_cnn(test_images[0],final_spot_dict)

    return final_spot_dict

def keras_model(weights_path):
    model = load_model(weights_path)
    return model

def img_test(test_image,final_spot_dict,model,class_dictionary):
    for i in range (len(test_images)):
        predicted_images = park.predict_on_image(test_images[i],final_spot_dict,model,class_dictionary)

def video_test(video_name,final_spot_dict,model,class_dictionary):
    name = video_name
    cap = cv2.VideoCapture(name)
    park.predict_on_video(name,final_spot_dict,model,class_dictionary,ret=True)

if __name__ == '__main__':
    test_images = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]
    weights_path = 'car1.h5'
    video_name = 'parking_video.mp4'
    class_dictionary = {}
    class_dictionary[0] = 'empty'
    class_dictionary[1] = 'occupied'
    park = Parking()
    park.show_image(test_images)
    final_spot_dict = img_process(test_images, park)
    model = keras_model(weights_path)
    img_test(test_images,final_spot_dict,model,class_dictionary)
    video_test(video_name,final_spot_dict,model,class_dictionary)

