import cv2
import os
import math
import numpy as np
import copy
import time
import timeout_decorator
from scipy import stats
import PIL
import json
from PIL import Image
import re
import base64
import shutil
from PIL import Image, ImageDraw, ImageFont
import os.path as osp
import fitz
import datetime
import signal
import yaml
import inspect
import requests
from tqdm import tqdm
from collections import defaultdict
from typing import Union, Optional, Any, List, Dict
import pandas as pd
from pprint import pprint
import dill
import random
import glob
import logging
from os.path import join as ospj
import hashlib
import os


cur_dir = os.path.dirname(__file__)
file_path = os.path.dirname(__file__)
# font = ImageFont.truetype( osp.join(cur_dir, "./fixed_files/simsun.ttf"),
#     30, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小

current_year = datetime.datetime.now().year
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
ten_years_l = []
for i in range(10):
    ten_years_l.append(str(current_year - i))
ten_years = '|'.join(ten_years_l)

year_re = re.compile(
    r'(.*?)({})(0[1-9]|1[0-2]|[1-9])?'.format(ten_years))
month_re = re.compile(r'(.*?)(0[1-9]|1[0-2]|[1-9])([0][1-9]|[1-2][0-9]|3[0-1]|[1-9])?')
day_re = re.compile(r'(.*?)([0][1-9]|[1-2][0-9]|3[0-1]|[1-9])')
date_re = re.compile(
    r'(.*?)({})(0[1-9]|1[0-2])([0][1-9]|[1-2][0-9]|3[0-1])'.format(ten_years))
date_weak_re = re.compile(
    r'(.*?)({})(.)(0[1-9]|1[0-2])(.)([0][1-9]|[1-2][0-9]|3[0-1])(\D)'.format(ten_years))
date_re_dict = {0: year_re, 1: month_re, 2: day_re}
WHITE_COLOER = (255, 255, 255)

iobs_test_api = 'http://ehis-ips-ai-stg.paic.com.cn/svc/api/iobs/upload/'


def timeOUT_judge(function,limited_time):
    """
    function:判断输入的函数是否运行超时，若超时抛出signal:'fail',未超时抛出：'success'.
    example:
        def test_function():
            print('funny!')
            time.sleep(3)
        print(util_.timeOUT_judge(test_function,2))
    """
    @timeout_decorator.timeout(limited_time, use_signals=False)
    def if_timeOUT(function):
        function()
        return 'success'
    try:
        return if_timeOUT(function)
    except timeout_decorator.TimeoutError:
        return 'fail'

class SplitInfo(object):
    def __init__(self, dicts, words, score):
        self.dicts = dicts
        middle_index = len(dicts)//2
        self.dict = dicts[:2]+dicts[middle_index-2:middle_index+2]+dicts[-2:]
        self.words = words
        self.score = round(score, 4)
        #logger.info('score:{}'.format(self.score))

def show(img, scale, windows=1):
    """
    show images
    :param img:
    :param scale:
    :param windows:
    :return:
    """
    assert hasattr(img, 'shape'), 'error image!'
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width*scale), int(height*scale)))

    # cv2.namedWindow('im', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('im',cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('im', windows)
    cv2.imshow('im', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def line_func(point1,point2,k):
    """
    get a line function by 2 points
    :param point1:
    :param point2:
    :param k: avoid denominator is 0
    :return:
    """
    point1 = list(point1)
    point2 = list(point2)
    def func(x):
        try:
            if point1[0]==point2[0]:
                point11=copy.deepcopy(point1)
                point11[0]+=1
                # print('point11[0]=',point11[0])
            else:
                point11 = copy.deepcopy(point1)
            return ((point11[1] - point2[1])/(point11[0]-point2[0])+k)*(x-point11[0])+ point11[1]
        except:     
            print('functiong error!!!!!!!!')
            print(point1,point2)
            raise Exception('Create function is error!!!')
    return func

def dis_points(point1,point2):
    """
    get the distance of 2 points.
    :param point1:
    :param point2:
    :return:
    """
    return math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)

def creat_decimal_array(a,b,c):
    """
    for example:
    creat_decimal_array(0,1,0.1)
    output: [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
    :param a:
    :param b:
    :param c:
    :return:
    """
    return np.arange(a,b,c)

def interpolation(point1,point2):
    """
    get more points between 2 points, the distance of the adjacent points is not over 1
    :param point1:
    :param point2:
    :return: a list including a series of points.
    """
    line_function = line_func(point1, point2, 0)
    l = [point1]
    for i in range(point1[0] + 1, point2[0]):
        l.append([i, round(line_function(i))])
    l.append(point2)
    #对Y方向插值
    l_copy=copy.deepcopy(l)
    for i in range(len(l_copy)-1):
        tmp=[]
        index=0
        if l_copy[i+1][1]-l_copy[i][1]>1:
            # index=1
            cmp=l_copy[i]
            while l_copy[i+1][1]-cmp[1]>1:
                cmp=[cmp[0],cmp[1]+1]
                tmp.append([cmp[0],cmp[1]])

            index = l.index(l_copy[i])
            l = l[:index+1] + tmp + l[index + 1:]
        if l_copy[i][1]-l_copy[i+1][1]>1:
            # index=1
            cmp = l_copy[i+1]
            while l_copy[i][1] - cmp[1] > 1:
                cmp = [cmp[0], cmp[1] + 1]
                tmp.append([cmp[0], cmp[1]])
            tmp=tmp[::-1]
            index = l.index(l[i])
            l = l[:index + 1] + tmp + l[index + 1:]
    return l

def find_circle(points,im):
    """
    find circumcircle  by some points
    :param points:
    :param im:
    :return:
    """
    l_points = []
    x_mean = 0
    y_mean = 0
    l_dis = []
    for i in range(len(points) // 2):
        if i == len(points) // 2 - 1:
            x_mean += points[i*2][0][0]
            y_mean += points[i*2][0][1]
            # l_points.append([points[i * 2], points[i * 2 + 1]])
            # l_dis.append(util.dis_points((points[i * 2], points[i * 2 + 1]), (points[0], points[1])))
            cv2.line(im, (points[i*2][0][0],points[i*2][0][1]), (points[0][0][0],points[0][0][1]), [255, 255, 255])
        else:
            # print(1. 验证矫正身份证签发机关识别结果算法的有效性
            # 2. 跟踪发票号识别错误率低的问题type(points[i*2]))
            x_mean += points[i*2][0][0]
            y_mean += points[i*2][0][1]
            l_points.append([points[i * 2], points[i * 2 + 1]])
            # l_dis.append(util.dis_points((points[i * 2], points[i * 2 + 1]), (points[i * 2 + 2], points[i * 2 + 3])))
            cv2.line(im, (points[i*2][0][0],points[i*2][0][1]), (points[i*2][0][0],points[i*2][0][1]), [255, 255, 255])
    # print('points=', points)
    # print('l_dis=', l_dis)
    x_mean = x_mean // (len(points) // 2)
    y_mean = y_mean // (len(points) // 2)

    l_tmp = copy.deepcopy(l_points)
    l_points = np.array(l_points)
    l_points = l_points.reshape((-1, 1, 2))
    # print('l_points=',l_points)
    # print('l_points.shape=', l_points.shape)

    (x, y), radius = cv2.minEnclosingCircle(l_points)
    # print('cv2.contourArea(contours[i])=', cv2.contourArea(l_points))
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(im, center, radius, (0, 255, 0), 2)
    cv2.circle(im, center, radius // 15, (0, 0, 255), 2)
    line_function = line_func(center, (x_mean, y_mean), i)
    points3 = (int(x_mean * 0.8), int(line_function(x_mean * 0.8)))
    points4 = (int(x_mean * 1.2), int(line_function(x_mean * 1.2)))
    cv2.line(im, points3, tuple(map(lambda x: x * 1, center)), (255, 0, 0), 2)
    cv2.line(im, points4, tuple(map(lambda x: x * 1, center)), (255, 0, 0), 2)

    show(im, 1)

def some_points_interpolation(points):
    """
    get some points
    :param points:
    :return:
    """
    points_tmp=[]
    for i in range(len(points) // 2):
        points_tmp.append([points[i*2],points[i*2+1]])
    points=points_tmp
    points_copy = copy.deepcopy(points)
    for i in range(len(points)):
        if i== len(points)-1:
            point1=points[i]
            point2=points[0]
        else:
            point1 = points[i]
            point2 = points[i+1]
        # print(point1,point2)
        # print('dis_points(point1,point2)=',dis_points(point1,point2))
        if dis_points(point1,point2)>=2:
            # print('point1,point2_dis=',point1,dis_points(point1,point2))
            # line_function=line_func((points[i*2],points[i*2+1]),(points[i*2+2],points[i*2+3]))
            if point1[0]<point2[0]:
                l=interpolation(point1,point2)
                # print('point1_error=', point1)
                # print('points_copy=',points_copy)
                index=points_copy.index(point1)

                points_copy=points_copy[:index]+l+points_copy[index+2:]
            else:
                l=interpolation(point2,point1)
                l=l[::-1]
                index = points_copy.index(point1)
                points_copy = points_copy[:index] + l + points_copy[index + 2:]
    l_tmp=[]
    for i in points_copy:
        l_tmp.append(i[0])
        l_tmp.append(i[1])
    # print('points_final=',l_tmp[16*2:21*2])

    return l_tmp[:-2]

def findCenter(l_points):
    """
    get the center of some points.
    :param l_points:
    :return:
    """
    l_points = np.array(l_points)
    l_points = l_points.reshape((-1, 1, 2))
    # print('l_points=',l_points)
    (x, y), radius = cv2.minEnclosingCircle(l_points)
    # print('cv2.contourArea(contours[i])=', cv2.contourArea(l_points))
    center = (int(x), int(y))
    return center

def simple_show(l_points, im, ishow=True, x_mean=0, y_mean=0, center=None, color=(255,0,0)):

    if isinstance(l_points,list):
        l_points = np.array(l_points)
        l_points = l_points.reshape((-1, 1, 2))
        # print('l_points=',l_points)

        if center==None:
            (x, y), radius = cv2.minEnclosingCircle(l_points)
            # print('cv2.contourArea(contours[i])=', cv2.contourArea(l_points))
            center = (int(x), int(y))
            radius = int(radius)
            # cv2.circle(im, center, radius, (0, 255, 0), 2)
            # cv2.circle(im, center, radius // 15, (0, 0, 255), 2)
        else:
            pass
            # cv2.circle(im, center, 3, (0, 0, 255), 2)

        if ishow:
            show(im,1)

        line_function = line_func(center, (x_mean, y_mean), 0)
        points3 = (int(x_mean * 0.8), int(line_function(x_mean * 0.8)))
        points4 = (int(x_mean * 1.2), int(line_function(x_mean * 1.2)))
        # cv2.line(im, points3, tuple(map(lambda x: x * 1, center)), color, 2)
        # cv2.line(im, points4, tuple(map(lambda x: x * 1, center)), color, 2)
        # cv2.line(im, (x_mean, y_mean), tuple(map(lambda x: x * 1, center)), (255, 0, 0), 2)

        # cv2.circle(im,(x_mean,y_mean),2,(0,0,255),3)

        if ishow:
            show(im, 1)
        return center
    elif isinstance(l_points, np.ndarray):
        sum1=len(l_points)
        # print(l1)
        for i in range(sum1):
            if i == sum1 - 1:
            #     print('l_points[i][0][0]=',l_points[i][0][0])
                cv2.line(im, (l_points[i][0][0], l_points[i][0][1]), (l_points[0][0][0], l_points[0][0][1]), color, 1)
            else:
                # print(l_points[i])

                cv2.line(im, (l_points[i][0][0], l_points[i][0][1]), (l_points[i + 1][0][0], l_points[i + 1][0][1]), color, 1)
                # cv2.circle(im,(l1[i*2][0][0],l1[i*2][0][1]),2,(0,255,255))

        if ishow:
            show(im, 1)
        # find_circle(l_points, im)

def draw_lines(points,im,close=True,color=(0,0,255)):
    """

    :param points: [[23,515],[514,524],...]
    :param im:
    :param close:
    :param color:
    :return:
    """
    if isinstance(points[0],list):
        for i in range(len(points)):
            if i < len(points)-1:
                cv2.line(im, (int(points[i][0]), int(points[i][1])), (int(points[i + 1][0]), int(points[i + 1][1])), color,
                         2)

            elif (i==len(points)-1) & close:
                cv2.line(im, (int(points[i][0]), int(points[i][1])), (int(points[0][0]), int(points[0][1])), color,
                         2)
            else:
                pass
    else:
        pass
    return im

#old name=extension_line
def draw_line(points1,points2,im):
    cv2.line(im, points1, points2, (255, 0, 0), 2)
    return im

def merge_imgs(leftgray,rightgray): # todo: 深入理解
    hessian = 400
    surf = cv2.xfeatures2d.SURF_create(hessian)  # 将Hessian Threshold设置为400,阈值越大能检测的特征就越少
    kp1, des1 = surf.detectAndCompute(leftgray, None)  # 查找关键点和描述符
    kp2, des2 = surf.detectAndCompute(rightgray, None)

    FLANN_INDEX_KDTREE = 0  # 建立FLANN匹配器的参数
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 配置索引，密度树的数量为5
    searchParams = dict(checks=50)  # 指定递归次数
    # FlannBasedMatcher：是目前最快的特征匹配算法（最近邻搜索）
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)  # 建立匹配器
    matches = flann.knnMatch(des1, des2, k=2)  # 得出匹配的关键点
    if len(matches)==0:
        return np.hstack((leftgray, leftgray))
    good = []
    # 提取优秀的特征点
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # 如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
            good.append(m)

    src_pts = np.array([kp1[m.queryIdx].pt for m in good])  # 查询图像的特征描述子索引
    dst_pts = np.array([kp2[m.trainIdx].pt for m in good])  # 训练(模板)图像的特征描述子索引
    print(src_pts,dst_pts)
    print(src_pts.shape,dst_pts.shape)
    H = cv2.findHomography(src_pts, dst_pts)  # 生成变换矩阵

    h, w = leftgray.shape[:2]
    h1, w1 = rightgray.shape[:2]
    shft = np.array([[1.0, 0, w], [0, 1.0, 0], [0, 0, 1.0]])
    M = np.dot(shft, H[0])  # 获取左边图像到右边图像的投影映射关系
    dst_corners = cv2.warpPerspective(leftgray, M, (w * 2, h))  # 透视变换，新图像可容纳完整的两幅图
    # cv2.imshow('tiledImg1', dst_corners)  # 显示，第一幅图已在标准位置
    dst_corners[0:h, w:w * 2] = rightgray  # 将第二幅图放在右侧
    return  dst_corners

def list2coordinate(points):
    """
    transform [x1,y1,x2,y2,x3,y3,x4,y4] into [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    :param points:
    :return:
    """
    l_points=[]
    for i in range(len(points) // 2):
        l_points.append([points[i * 2], points[i * 2 + 1]])
    return l_points

def coordinate2list(points):
    #input:points=[[x1,y1,x2,y2,x3,y3,x4,y4],[x1',y1',x2',y2',x3',y3',x4',y4'],...]
    #output:l_points=[x1,y1,x2,y2,x3,y3,x4,y4,x1',y1',x2',y2',x3',y3',x4',y4',...]
    l_points=[]
    for i in range(len(points)):
        # print('points[i]=',points[i])
        for j in points[i]:
            if isinstance(j[0],str):
                l_points.append(int(j[0]))
                l_points.append(int(j[1]))
            else:
                l_points.append(j[0])
                l_points.append(j[1])
    return l_points



def take_boxes_from_xml(path_tmp,get_label=False):
    f = open(path_tmp)
    lines = f.readlines()
    xmin_index = []
    for index, item in enumerate(lines):
        if ('xmin' in item):
            xmin_index.append(index)
    print('xmin_index: ', xmin_index)
    rectangle = []
    for i in range(len(xmin_index)):
        box_context = lines[xmin_index[i]: xmin_index[i] + 4]
        single_coordinate_points = get_boxes_from_xml(
            box_context)
        if get_label:
            nameLine=lines[xmin_index[i]-5]
            single_coordinate_points.append(nameLine.split('>')[1].split('<')[0])
            # rectangle.append(single_coordinate_points)
            # continue
        rectangle.append(single_coordinate_points)
    return rectangle

def getFourPoints(box):
    # function:按左上，右上，右下，左下排列坐标
    # input:box=[x1,y1,x2,y2,x3,y3,x4,y4]
    # output:box=[x1,y1,x2,y2,x3,y3,x4,y4]
    #or:
    # input:box=[x1,y1,x2,y2,x3,y3,x4,y4,confidence]
    # output:box=[x1,y1,x2,y2,x3,y3,x4,y4,confidence]
    index=0
    # print('box0====', box)
    if len(box)%2==1:
        index=1
        confidence = box[-1]
        box=box[:-1]

    # if len(box)==8:
    #     box=list2coordinate(box)
    # print('box====',box)
    if abs(box[0]-box[2])>abs(box[1]-box[3]):
        points_y_part1=[[int(box[0]),int(box[1])],[int(box[2]),int(box[3])]]
        points_y_part2=[[int(box[4]),int(box[5])],[int(box[6]),int(box[7])]]
    else:
        points_y_part1 = [[int(box[0]), int(box[1])], [int(box[6]), int(box[7])]]
        points_y_part2 = [[int(box[2]), int(box[3])], [int(box[4]), int(box[5])]]
    if (points_y_part1[0][1]+points_y_part1[1][1])>(points_y_part2[0][1]+points_y_part2[1][1]):
        points_y_part1,points_y_part2=points_y_part2,points_y_part1
    points_y_part1=sorted(points_y_part1,key=lambda x:x[0])
    points_y_part2=sorted(points_y_part2,key=lambda x:x[0])
    if index==0:
        return points_y_part1[0]+points_y_part1[1]+points_y_part2[1]+points_y_part2[0]
    else:
        return points_y_part1[0] + points_y_part1[1] + points_y_part2[1] + points_y_part2[0]+[confidence]

def order_4points(coordinate):
    # function:按左上，右上，右下，左下排列坐标
    for i in range(len(coordinate)):
        coordinate[i]=getFourPoints(coordinate[i])
    return coordinate

def iou_y(box1,box2):
    if box1[0]>box2[0]:
        box1,box2=box2,box1
    # coordinate_lt,+coordinate_rt,coordinate_rd,coordinate_ld=getFourPoints(box1)
    # coordinate2_lt, coordinate2_ld, coordinate2_rt, coordinate2_rd=getFourPoints(box2)
    iou_y_dis=min(box1[5],box2[7])-max(box1[3],box2[1])
    if iou_y_dis <= 0:
        return 0
    return max(iou_y_dis/(box1[5]-box1[3]),iou_y_dis/(box2[7]-box2[1]))

def iou_x(box1,box2):
    # coordinate1_lt, coordinate1_ld, coordinate1_rt, coordinate1_rd = getFourPoints(box1)
    # coordinate2_lt, coordinate2_ld, coordinate2_rt, coordinate2_rd = getFourPoints(box2)
    iou_y_dis=min(box1[4],box2[4])-max(box1[0],box2[0])
    if iou_y_dis <= 0:
        return 0
    return max(iou_y_dis/(box1[4]-box1[0]),iou_y_dis/(box2[4]-box2[0]))


def include_angle(points1,points2=None): #input points example:[1,1,3,3] or [(1,1),(3,3)]
    """
    get anble  between two lines or one lines and x axis
    :param points1:
    :param points2:
    :return:
    """
    def angle(v1, v2):
        dx1 = v1[2] - v1[0]
        dy1 = v1[3] - v1[1]
        dx2 = v2[2] - v2[0]
        dy2 = v2[3] - v2[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(angle1 * 180 / math.pi)
        angle2 = math.atan2(dy2, dx2)
        angle2 = int(angle2 * 180 / math.pi)
        if angle1 * angle2 >= 0:
            included_angle = abs(angle1 - angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle

    def angle_oneline(v):
        dx = v[2] - v[0]
        dy = v[3] - v[1]
        angle = math.atan2(dy, dx)
        angle = angle * 180 / math.pi
        if angle < 0:
            angle = 360 - abs(angle)
        return angle

    def transInput(points):
        if len(points) == 2:
            tmp = []
            for i in points:
                tmp.append(i[0])
                tmp.append(i[1])
            points=tmp
        return points

    def main(points1,points2):
        points1=transInput(points1)
        if points2!=None:
            points2 = transInput(points2)
            # horizontal = [0, 0, 1, 0]
            return angle(points1, points2)

        else:
            return angle_oneline(points1)

    return main(points1,points2)

def numpy_savetxt_or_loadtxt(save=True,file_path='',txt=0):
    if save==True:
        np.savetxt(file_path, txt)
    else:
        return np.loadtxt(file_path)

def dumpRotateImage(img, degree):
    """
    rotate image by any angle
    :param img:
    :param degree:
    :return:
    """
    height, width = img.shape[:2]
    heightNew = int(abs(width * math.fabs(math.sin(math.radians(degree)))) + abs(height * math.fabs(math.cos(math.radians(degree)))))
    widthNew = int(abs(height * math.fabs(math.sin(math.radians(degree)))) + abs(width * math.fabs(math.cos(math.radians(degree)))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0,2] += (widthNew - width) // 2
    matRotation[1,2] += (heightNew - height) // 2
    # imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue = (255, 255, 255))
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew))

    return imgRotation, matRotation


def FindPoints_for_after_rotate(points,matRotation):
    """
    map ori coordinate into new coordinate with rotated images
    :param points:
    :param matRotation:
    :return:
    """
    # points=[3,54,62,...]
    points_new = []
    for i in range(len(points) // 2):
        pt = np.dot(matRotation, np.array([[points[i * 2]], [points[i * 2 + 1]], [1]]))
        points_new.append(int(round(pt[0][0])))
        points_new.append(int(round(pt[1][0])))
    return points_new


def factorial(a, b): # a < b
    """
    get a*(a+1)*(a+2)*...*(b-1)*b
    :param a:
    :param b:
    :return:
    """
    if a == b:
        return a
    return b*factorial(a, b - 1)


def randome_take_b_from_a_probability(a,b): #a<b
    return factorial(b+1-a,b)*1.0/factorial(1,a)


def draw_line_for_4pointsBox(coordinate,im):
    """
    draw boxes im image
    :param coordinate: [[500, 66, 957, 57, 958, 111, 501, 120], [1084, 111, 1385, 106, 1386, 148, 1085, 154], ...]
    :param im: cv2 format
    :return:
    """
    for i in coordinate:
        for j in range(4):
            if j == 3:
                cv2.line(im, (int(i[j * 2]), int(i[j * 2 + 1])), (int(i[0]), int(i[1])), (0, 0, 255), 2)
                break
            # print('i=', i)
            cv2.line(im, (int(i[j * 2]), int(i[j * 2 + 1])), (int(i[(j + 1) * 2]), int(i[(j + 1) * 2 + 1])), (0, 0, 255), 2)
    # show(im,0.5)
    return im


# def drawTXT(im, text, points, color=(255, 0, 0)):
#     cv2img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
#     pilimg = Image.fromarray(cv2img)

#     # PIL图片上打印汉字
#     draw = ImageDraw.Draw(pilimg)  # 图片上打印
#     draw.text(points, text, color, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体

#     # PIL图片转cv2 图片
#     im = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
#     return im


def draw_boxes_and_texts(coordinate, texts, im):
    """
    draw boxes im image
    :param coordinate: [[500, 66, 957, 57, 958, 111, 501, 120], [1084, 111, 1385, 106, 1386, 148, 1085, 154], ...]
    :param im: cv2 format
    :return:
    """
    assert len(coordinate) == len(texts),  "the number of coordinate and texts don't match!"
    for i in coordinate:
        for j in range(4):
            font = cv2.FONT_HERSHEY_SIMPLEX
            if j == 3:
                cv2.line(im, (int(i[j * 2]), int(i[j * 2 + 1])), (int(i[0]), int(i[1])), (0, 0, 255), 2)
                break
            # print('i=', i)
            cv2.line(im, (int(i[j * 2]), int(i[j * 2 + 1])), (int(i[(j + 1) * 2]), int(i[(j + 1) * 2 + 1])), (0, 0, 255), 2)
    for idx, coordinate_sub in enumerate(coordinate):
        text_s = texts[idx]
        im = drawTXT(im, text_s, (coordinate_sub[0], coordinate_sub[1]-30))
    return im


def loadJson(json_path):
    #原始函数名为loadFont
    with open(json_path, encoding='utf-8')  as f: #设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错
        setting = json.load(f)
    # print('setting=',setting)
    all_points=[]
    for i in setting['shapes']:
        all_points.append(i['points'])
    # family = setting['shapes'][0]['points']   #/注意多重结构的读取语法
    # size = setting['fontSize']
    print('all_points=', all_points)
    return all_points

def take_mode(l):
    # 取列表的众数
    return stats.mode(l)[0][0]

def take_median(l):
    # 取列表的中位数
    return np.median(np.array(l))


def PIL_rotate():
    img = PIL.Image.open('/Users/youjincheng749/Documents/images/timg.jpeg').convert('RGB')
    img_alpha = img.convert('RGBA')
    # ~W~K转~[~C~O
    angle=0.5
    rot = img_alpha.rotate(angle, expand=1)
    #print(rot.size[1])
    # ~N~W~K转~[~C~O大~O~[~P~L~Z~D~Y~I~L~_~_
    fff = PIL.Image.new('RGBA', rot.size, (255, 255, 255, 255))
    # # 使~Trot~\为~H~[建~@个~M~P~H~[~C~O
    out = PIL.Imaage.composite(rot, fff, mask=rot)
    out.show()


def ifnot2create(name):
    if isinstance(name,list):
        for i in name:
            if not os.path.exists(i):
                os.mkdir(i)
    else:
        if not os.path.exists(name):
            os.mkdir(name)


def iou_vertical(box1, box2):
    if box1[0] > box2[0]:
        box1, box2 = box2, box1
    iou_y_dis = min(box1[5], box2[7])-max(box1[3], box2[1])
    if iou_y_dis <= 0:
        return 0
    return max(iou_y_dis/(box1[5]-box1[3]), iou_y_dis/(box2[7]-box2[1]))


def iou_horizontal(box1, box2):
    iou_y_dis = min(box1[4], box2[4])-max(box1[0], box2[0])
    if iou_y_dis <= 0:
        return 0
    return max(iou_y_dis/(box1[4]-box1[0]), iou_y_dis/(box2[4]-box2[0]))


def get_digits(s):
    l = list(s)
    tmp = []
    for i in l:
        if i.isdigit():
            tmp.append(i)
    return ''.join(tmp)


def plt2opencv(image):
    # image = Image.open("plane.jpg")
    # image.show()
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return image


def opencv2plt(img):
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return image


def get_boxes_from_xml(box_context):
    #original name: cal_single_coordinate_points
    xmin = int(box_context[0].split(">")[1].split("<")[0])
    ymin = int(box_context[1].split(">")[1].split("<")[0])
    xmax = int(box_context[2].split(">")[1].split("<")[0])
    ymax = int(box_context[3].split(">")[1].split("<")[0])

    coor = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
    return coor


def read_txt_lines(txt_path):
    with open(txt_path, "r") as w:
        return w.readlines()


def read_txt_all(txt_path):
    with open(txt_path, "r") as w:
        return w.read()



def load_txt(txt_path, key = '', label = False, split_s = ','):
    
    if txt_path.split('.')[-1] != 'txt':
        raise TypeError("Only supply txt file!")
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_keys_from_txt(txt_path):
    _, names_all = load_txt(txt_path, label=True)
    return list(set(names_all))


def get_split(txt_path, isXml=False):
    if isXml:
        coordinate=take_boxes_from_xml(txt_path, addOcr=True)
    else:
        coordinate=load_txt(txt_path)
    if coordinate:
        splits = [SplitInfo(getFourPoints(item[1:]), item[0].strip('\n'), 0) for item in coordinate]
        return splits
        # return [SplitInfo(getFourPoints(item[1:]), item[0], 0) for item in coordinate]

def is_Chinese(word):
    """
    Function: judge whether the single char is chinese
    :return: True or False
    """
    if '\u4e00' <= word <= '\u9fff':
        return True
    return False


def get_most_ext(file_names):
    """
    Function: get the ext that appears most
    :param file_names: a list including some file names with ext
    :return: the ext that appears most in file_names
    """
    # print('file_names: ', file_names)
    exts_num = {}
    most_num_ext = ''
    most_num = 0
    for file_name in file_names:
        ext = file_name.split('.')[-1]
        if ext not in exts_num:
            exts_num[ext] = 1
        else:
            exts_num[ext] += 1
        if exts_num[ext] > most_num:
            most_num = exts_num[ext]
            most_num_ext = ext
    return most_num_ext.strip('\n').strip()


def compute_iou(rec1_ori, rec2_ori):
    """
    computing IoU
    param rec1, rec2: (xmin, ymin, xmax, ymax)
    """
    rec1 = copy.deepcopy(rec1_ori)
    rec2 = copy.deepcopy(rec2_ori)
    if len(rec1) == 8:
        rec1[0], rec1[2], rec1[1], rec1[3] = get_min_max(rec1)
        # print('rec2: ', rec2)
        rec2[0], rec2[2], rec2[1], rec2[3] = get_min_max(rec2)
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return float(intersect) / float(sum_area - intersect)


def write2txt(save_name, contents):
    """
    Function: write all type variables into txt file.
    """
    if not isinstance(contents, str):
        contents = str(contents)
    with open(save_name, "w") as w:
        w.write(contents)


def aline(s='*'):
    if not isinstance(s, str):
        s = str(s)
    print(s * 66)


def get_min_max(coordinate):
    if len(coordinate) % 2 != 0:
        print('the number of coordinate is not even, but the number is ' + str(len(coordinate)))
        raise 'the number of coordinate is not even, but the number is ' + str(len(coordinate))

    x_min = min(coordinate[::2])
    x_max = max(coordinate[::2])
    y_min = min(coordinate[1::2])
    y_max = max(coordinate[1::2])

    return x_min, x_max, y_min, y_max


def rm_punctuation(line):
    rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
    line = rule.sub('',line)
    return line


def rm_Chinese(word):
    # print('|', word, '|')
    new_word = ''
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            continue
        new_word += ch
    # print('new_word: ', new_word)
    return new_word


def get_base64_data(img_path):
    # 参数image：图像base64编码
    with open(img_path, 'rb') as f:
        img = base64.b64encode(f.read())
    return img


def save_json(info, json_path):
    with open(json_path, 'w')as f:
        json.dump(info, f)


def read_json(json_path, encoding_type='utf-8'):
    with open(json_path, 'r', encoding=encoding_type) as f:
        str_ = f.read()
        if not str_.strip(' '):
            return
        return json.loads(str_)


def mkdir_dir(new_dir):
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)


def rm_garbage(files_dir):
    names = os.listdir(files_dir)
    for name in names:
        if '.DS' in name or '.ipynb_checkpoints' in name:
            # print(name)
            # os.imgs_txt(os.path.join(dir_path, name))
            filepath = os.path.join(files_dir, name)
            shutil.rmtree(filepath, True)


def digit_num(s):
    num = 0
    for i in s:
        if i.isdigit():
            num += 1
    return num


def split_str_by_character(t):
    t = re.split(r"[\D]+", t)
    return t


def get_series_char_max_num(str1, str2):
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]  # 多一位
    maxNum = 0  # 最长匹配长度
    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                # 相同则累加
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > maxNum:
                    # 获取最大匹配长度
                    maxNum = record[i + 1][j + 1]
    return maxNum


split_by_punctuation_re = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]+")
def split_by_punctuation(s):
    return split_by_punctuation_re.split(s)


def get_antonym_word():

    def update_dict(a, b, result_d):
        """
        Add {a:b} into result_d
        :param a: key
        :param b: value, type(b) = list
        :param result_d: dict
        :return:
        """
        if a in result_d:
            if b not in result_d[a]:
                result_d[a].append(b)
        else:
            result_d[a] = [b]

    lines = read_txt_lines('反义词库.txt')
    lines = [i.strip('\n') for i in lines]
    # print('lines: ', lines)
    lines = set(lines)
    result_d_l = [{}, {}, {}]
    for i in lines:
        l = split_by_punctuation(i)
        # aline()
        # print('l: ', l)
        if len(l[0]) > 2:
            continue
        if len(l[0]) == 1:
            update_dict(l[0], l[1], result_d_l[-1])
            update_dict(l[1], l[0], result_d_l[-1])
        elif len(l[0]) == 2:
            update_dict(l[0], l[1], result_d_l[-2])
            update_dict(l[1], l[0], result_d_l[-2])

    return result_d_l


def get_negative_word():
    lines = read_txt_lines('否定词库.txt')
    lines = [i.strip('\n') for i in lines]
    lines = set(lines)
    result_s = set()
    for i in lines:
        l = split_by_punctuation(i)
        if len(l[0]) > 2:
            continue
        if l[0] not in result_s:
            result_s.add(l[0])
    return result_s


def copy_files_by_name(name_l, ori_path, new_path):
    mkdir_dir(new_path)
    for name in name_l:
        name_ori_path = osp.join(ori_path, name)
        name_new_path = osp.join(new_path, name)
        shutil.copy(name_ori_path, name_new_path)


def edit_distance(word1, word2):
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta,
                           min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]


def has_file_allowed_extension(filename, extensions):
    """查看文件是否是支持的可扩展类型

    Args:
        filename (string): 文件路径
        extensions (iterable of strings): 可扩展类型列表，即能接受的图像文件类型

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions) # 返回True或False列表


def write_csv(l, csv_name):
    with open(csv_name, 'w') as f:
        for i in l:
            f.write(','.join(i))
            f.write('\n')


def judge_empty_file(txt_path):
    true_str = read_txt_lines(txt_path)
    for i in true_str:
        i = i.replace(' ', '').replace('  ', '').strip('\n')
        if i and i != '\ufeff':
            return False
    return True


def move_empty_txt(true_txt_dir):
    empty_dir = true_txt_dir + '_empty'
    if not osp.exists(empty_dir):
        os.mkdir(empty_dir)
    names = os.listdir(true_txt_dir)
    for name in names:
        if not name.endswith('.txt'):
            continue
        empty_name = ''
        true_txt_path = osp.join(true_txt_dir, name)
        if judge_empty_file(true_txt_path):
            print('empty_name: ', empty_name)
            ori_path = osp.join(true_txt_dir, name)
            new_path = osp.join(empty_dir, name)
            # print(ori_path, new_path)
            shutil.move(ori_path, new_path)


def draw_middle_line(im, coordinate_l):
    """
    draw middle line for the box with 4 points
    :param im:
    :param coordinate_l:
    :return:
    """
    for coordinate in coordinate_l:
        left_points = (np.array(coordinate[:2]) + np.array(coordinate[-2:])) / 2
        right_points = (np.array(coordinate[2:4]) + np.array(coordinate[4:6])) / 2
        left_points = left_points.astype(np.int).tolist()
        right_points = right_points.astype(np.int).tolist()
        draw_line(tuple(left_points), tuple(right_points), im)


class rotateImgByCoordinate(object):
    """
    rotate image by text location coordinate
    """
    def get_middle_points(self):
        middle_points = []
        for coordinate in self.coordinate_l:
            left_points = (np.array(coordinate[:2]) + np.array(coordinate[-2:])) / 2
            right_points = (np.array(coordinate[2:4]) + np.array(coordinate[4:6])) / 2
            left_points = left_points.astype(np.int).tolist()
            right_points = right_points.astype(np.int).tolist()
            middle_points.append([left_points, right_points])
        return middle_points

    def get_correct_angle(self, all_angle_l):
        for idx, angle in enumerate(all_angle_l):
            if angle > 180:
                all_angle_l[idx] = angle - 360
        return all_angle_l

    def get_avg_angle(self):
        middle_points = self.get_middle_points()
        all_angle_l = []
        for middle_point in middle_points:
            sub_angle = include_angle(middle_point)
            all_angle_l.append(sub_angle)
        all_angle_l = self.get_correct_angle(all_angle_l)
        # print('all_angle_l: ', all_angle_l)
        # angle = utils_.take_median(all_angle_l)
        angle = np.mean(all_angle_l).item()
        return angle

    def get_new_points(self, matRotation):
        for idx, coordinate in enumerate(self.coordinate_l ):
            self.coordinate_l[idx] = FindPoints_for_after_rotate(coordinate, matRotation)

    def rotate_img(self, im, coordinate_l):
        self.coordinate_l = coordinate_l
        angle = self.get_avg_angle()
        print('angle: ', angle)
        # draw_middle_line(im, coordinate_l)
        # utils_.show(im, 0.5)
        show(im, 0.5)
        img_rotated, matRotation = dumpRotateImage(im, angle)
        show(img_rotated, 0.5)
        self.get_new_points(matRotation)
        return img_rotated, self.coordinate_l


def pdf2img(file_path, dest_path):
    zoom_x = 2.0  # horizontal zoom
    zomm_y = 2.0  # vertical zoom
    mat = fitz.Matrix(zoom_x, zomm_y)  # zoom factor 2 in each dimension
    doc = fitz.open(file_path)  # open document
    image_paths = []
    for page in doc:  # iterate through the pages
        pix = page.getPixmap(matrix=mat)
        name = os.path.basename(file_path)
        image_path = '{0}/{1}_{2}.jpg'.format(dest_path, name, page.number + 1)
        pix.writeImage(image_path)
        image_paths.append(image_path)
    return image_paths

def rectify_date(date):
    date_l = date.split('-')
    if len(date_l) == 3:
        if len(date_l[1]) == 1:
            date_l[1] = '0' + date_l[1]
        if len(date_l[2]) == 1:
            date_l[2] = '0' + date_l[2]

    return '-'.join(date_l)


def change_date_format(date):
    if '-' not in date:
        return f"{date[:4]}-{date[4:6]}-{date[6:]}"
    else:
        return date


def get_subimg(img, dicts, labels=[]):
    sub_imgs = []
    new_dicts = []
    n = len(dicts)
    idx = 0
    while idx < n:
        i = dicts[idx]
        Points = list2coordinate(i)

        x_min = min(np.array(Points)[:, 0])
        x_max = max(np.array(Points)[:, 0])
        y_min = min(np.array(Points)[:, 1])
        y_max = max(np.array(Points)[:, 1])
        new_Points = [[j[0]-x_min, j[1]-y_min] for j in Points]

        sub_img = img[y_min:y_max, x_min:x_max]
        if sub_img.size == 0:
            if labels:
                del labels[idx]
                del dicts[idx]
                n -= 1
            continue
        mask = np.ones(sub_img.shape, np.uint8) * 255
        mask1 = cv2.fillPoly(mask, [np.array(new_Points)], (0, 0, 0))
        roi = np.where(mask1 > sub_img, mask1, sub_img)

        sub_imgs.append(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)))
        new_dicts.append(i)
        idx += 1
    if labels:
        return sub_imgs, new_dicts, labels
    return sub_imgs, new_dicts


class ParseDate(object):
    def parse_date(self, str_, date_match):
        res = []
        date = date_match.match(str_)
        if date:
            date = date.groups()
            for j in date[1:]:
                if j: res.append(j)
        return res

    def get_date(self, t):
        m = date_re.match(t)
        if m:
            date_l = m.groups()[1:]
            return '-'.join(date_l)
        else:
            return None

    def get_weak_date(self, t):
        m = date_weak_re.match(t)
        if m:
            date_l = m.groups()[1::2]
            return '-'.join(date_l)
        else:
            return None

    def get_date_by_str(self):
        """通过空格、文字、'-'及其他字符切分后正则得到日期"""
        result = []  # date list
        parse_flag = 0  # 0->year, 1->month, 2->day, 3->over
        words_l = split_str_by_character(self.check_s)
        res = []
        for words in words_l:
            res.extend(self.parse_date(words, date_re_dict[parse_flag]))
            parse_flag = len(res)
            if parse_flag == 3:
                result.append('-'.join(res))
                break
        return result

    def get_date_by_re(self):
        """去除字符串中非数字字符后正则得到日期"""
        date_list = []
        words_l = split_str_by_character(self.check_s)
        for words in words_l:
            if self.get_date(words):
                date_list.append(self.get_date(words))
        return date_list

    def get_date_by_weak_re(self):
        """获取0000*00*00*格式的日期"""
        date_list = []
        if self.get_weak_date(self.check_s):
            date_list.append(self.get_weak_date(self.check_s))
        return date_list

    def get_date_list(self, check_s):
        date_list = []
        self.check_s = check_s
        date_list.extend(self.get_date_by_re())  # 去除字符串中非数字字符后正则得到日期
        date_list.extend(self.get_date_by_weak_re())  # 获取0000*00*00*格式的日期
        # print('date_list2: ', date_list)
        date_list.extend(self.get_date_by_str()) # 通过空格、文字、'-'及其他字符切分后正则得到日期
        # print('date_list1: ', date_list)
        return date_list


class SplitInfo(object):
    def __init__(self, dicts, words, score=1):
        self.dicts = dicts
        middle_index = len(dicts) // 2
        self.dict = dicts[:2]+dicts[middle_index-2:middle_index+2]+dicts[-2:]
        self.words = words
        self.score = round(score, 4)


# def draw_line_box(im, i):
#     """
#     im: image to draw.
#     i: coordicate list, like: [62, 1435, 163, 1435, 163, 1466, 62, 1466]
#     """
#     n = len(i)//2
#     for j in range(n):
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         if j==n-1:
#             cv2.line(im, (int(i[j * 2]), int(i[j * 2 + 1])), (int(i[0]), int(i[1])),(0,0,255),4)
#             break
#         # print('i=',i)
#         print(im.shape)
#         print((int(i[j * 2]), int(i[j * 2 + 1])), (int(i[(j + 1) * 2]), int(i[(j + 1) * 2 + 1])))
#         cv2.line(im, (int(i[j * 2]), int(i[j * 2 + 1])), (int(i[(j + 1) * 2]), int(i[(j + 1) * 2 + 1])),(0,0,255),4)
#     return im

def reverse_dict(d):
    res_d = {}
    for key, value in d.items():
        res_d[value] = key
    return res_d


def init_logger():
    logging.basicConfig(level=logging.DEBUG, format='%(filename)s:%(lineno)s|%(message)s')
    return logging


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right



class PaddingImgs(object):
    def get_pading_length(self, side1, side2):
        if abs(side1-side2)%2==0:
            pading1=abs(side1-side2)//2
            pading2=pading1
        else:
            pading1=abs(side1-side2)//2
            pading2=pading1+1
        return pading1, pading2
    
    def padding_imgs2same(self, img1, img2, orient='h', just='center'):
        print(img1.shape, img2.shape)
        if orient == 'h':
            shape_idx = 1
        elif orient == 'w':
            shape_idx = 0
        else:
            raise ValueError('illegal orient!')
        if img1.shape[shape_idx] == img2.shape[shape_idx]:
            return img1, img2
        pading1, padding2 = self.get_pading_length(img1.shape[shape_idx], img2.shape[shape_idx]) 
        processed_img = img1
        if img1.shape[shape_idx] > img2.shape[shape_idx]:
            processed_img = img2
        if orient == 'h':
            if just == 'center':
                constant = cv2.copyMakeBorder(processed_img, 0, 0, pading1, padding2, cv2.BORDER_CONSTANT, value=WHITE_COLOER)
            elif just == 'left':
                constant = cv2.copyMakeBorder(processed_img, 0, 0, 0, pading1+padding2, cv2.BORDER_CONSTANT, value=WHITE_COLOER)
            else:
                raise ValueError('just is invalid')
        else:
            if just == 'center':
                constant = cv2.copyMakeBorder(processed_img, pading1, padding2, 0, 0, cv2.BORDER_CONSTANT, value=WHITE_COLOER) 
            elif just == 'up':
                constant = cv2.copyMakeBorder(processed_img, 0, pading1+padding2, 0, 0, cv2.BORDER_CONSTANT, value=WHITE_COLOER) 
            else:
                raise ValueError('just is invalid')
        # print(img1.shape, img2.shape, constant.shape)
        if processed_img is img1:
            return constant, img2
        return img1, constant


padding_imgs = PaddingImgs()


def joint_imgs(img1, img2, mode='scale', orient='h', just='center'):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    if mode == 'scale':
        img1 = cv2.resize(img1, (int(h*w1/h1), h))
        img2 = cv2.resize(img2, (int(h*w2/h2), h))
    else:
        img1, img2 = padding_imgs.padding_imgs2same(img1, img2, orient, just)
    print(img1.shape, img2.shape)
    if orient == 'h':
        return np.vstack((img1, img2))
    elif orient == 'w':
        return np.hstack((img1, img2))


def get_current_date(formate='%Y_%m_%d'):
    f1 = datetime.date.today()
    return f1.strftime(formate)



def save_dict2yaml(file_path, d):
    with open(file_path, 'w') as f:
        data = yaml.dump(d, f)


def read_from_yml(yml_path):
    with open(yml_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def meger_more_txt2one(txts_dir):
    result_d = {}
    for name in os.listdir(txts_dir):
        if not name.endswith('.txt'):
            continue
        result_d[name] = read_txt_all(osp.join(txts_dir, name))
    save_txt = osp.join(txts_dir, 'result.yml')
    if osp.isfile(save_txt):
        raise "save_txt'name conficts with origin_txt's"
    save_dict2yaml(save_txt, result_d)
        
    
# def retrieve_name(var):
#     callers_local_vars = inspect.currentframe().f_back.f_locals.items()
#     return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def retrieve_name(obj, namespace):
     return [name for name in namespace if namespace[name] is obj]

def reverse_variate_in_list_2str(l_s, split_s=','):
    l_s = l_s.strip('[').strip(']')
    result_l = l_s.split(split_s)
    return result_l


def get_real_list(s, split_s=' '):
    ori_l = reverse_variate_in_list_2str(s, split_s)
    result_l = []
    for idx, elem in enumerate(ori_l):
        elem = elem.strip('\n').strip()
        if not is_float(elem):
            continue
        result_l.append(float(elem))
    return result_l


def is_float(s):
    s = str(s)
    if s.count('.') ==1:
        left = s.split('.')[0]
        right = s.split('.')[1]
        if right.isdigit():
            if left.count('-')==1 and left.startswith('-'):
                num = left.split['-'][-1]
                if num.isdigit():
                    return True
            elif left.isdigit():
                return True
    return False


def print_more_variate_type(spacename, *arge):
    for i in arge:
        print(
            retrieve_name(i, spacename)[0], type(i), sep=': '
        ) 


def numpy2list(n, decimals=1):
    n = np.around(n, decimals=decimals)
    return n.tolist()


def get_current_file_dir(__file__):
    return os.path.dirname(__file__)


def get_file_size(file_path):
    file_size = os.stat(file_path).st_size / 1000 / 1000
    return file_size


def image_compression(im_path, target_m = 5, need_path = False):
    """
    图像压缩到5m以内
    :param file_path:
    :return:
    """
    img = cv2.imread(im_path)
    new_im_path = os.path.splitext(im_path)[0]+'_compression.jpg'
    quality = 95
    compressed_flag = False
    count_n = 0
    while quality > 0:
        cv2.imwrite(new_im_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        file_size = os.stat(new_im_path).st_size / 1024 / 1024
        print('file_size: ', file_size)
        if file_size <= target_m:
            break
        quality -= 10 if file_size >= 6.5 else 5
        # quality -= 30 if file_size >= 1 else 5   # 图像大小大于6.5M时以-10衰减，否则以5衰减
        # quality -= np.clip((file_size - target_m) * 10, 2, 30)
        # quality -= np.clip((file_size - target_m) * 60, 2, 30)
        # quality -= np.clip((file_size - target_m) * 60, 2, 15)
        # quality -= (target_m - file_size)**2
        # quality -= max(1, ((target_m - file_size) * 10 / 5)**2 * 10)
        print('quality: ', quality)
        compressed_flag = True
        count_n += 1
    print('count_n: ', count_n)
    # file = open(new_im_path, 'rb')
    if compressed_flag:
        file = open(new_im_path, 'rb')
    else:
        file = open(im_path, 'rb')
    
    # os.system('rm {}'.format(new_im_path))   # itg最后以案件号为单位删除
    if need_path:
        if not compressed_flag:
            new_im_path = im_path
        return file, new_im_path
    return file


def iter_files(dst_dir):
    for root, dirs, files in os.walk(dst_dir):
        for file in files:
            file_path = os.path.join(root, file)
            yield file_path


def upload_iobs(file_path: str) -> str:
    """upload image into IOBS and get its url.
    Args:
        file_path: the file path
    Returns:
        iobs_url: the iobs url corresponding to the given file path.
    """
    formdata = {"appId": "ehis_ips"}
    name = osp.basename(file_path)
    files = {"file": (name, open(file_path, "rb"))}
    response = requests.post(iobs_test_api, data=formdata, files=files)
    if not json.loads(response.text).get('data'):
        return ''
    iobs_url = json.loads(response.text).get('data').get('url')
    return iobs_url


def get_name_no_ext(name):
    l = name.split('.')[:-1]
    return '.'.join(l)


def get_files_number(dst_dir: str, ext_l: Union[list, str, None]=None) -> int:
    res = 0
    if isinstance(ext_l, str): 
        ext_l = [ext_l]
    for root, _, files in os.walk(dst_dir):
        for file in files:
            if ext_l and not file.split('.')[-1] in ext_l:
                    continue
            res += 1
    return res


def find_empty_file(file_dir: str) -> None:
    """judge whether the empty file exits, if exiting, return True
    """
    n = get_files_number(file_dir)
    with tqdm(total=n) as pbar:
        pbar.set_description('Processing: ')
        for root, _, files in os.walk(file_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if judge_empty_file(file_path):
                    print(f'find empty file: {file}, stop search!')
                    return True
                pbar.update(1)
        return False

def find_same_name(dst_dir: str) -> dict:
    """find whether different files in different dirs has the same name.
    """
    cache_d, res_d = defaultdict(list), defaultdict(list)
    n = get_files_number(dst_dir)
    with tqdm(total=n) as pbar:
        pbar.set_description('Processing: ')
        for root, _, files in os.walk(dst_dir):
            for file in files:
                cache_d[file].append(root)
                if len(cache_d[file]) > 1:
                    res_d[file] = cache_d[file]
                pbar.update(1)
    return res_d
                

def check_file(file_name: str) -> bool:
    """check whether file is legel, if legel, return True, else return False.
    """
    if '.DS_Store' in file_name or file_name.split('.')[0] == '' or '-checkpoint.' in file_name:
        return False
    else:
        return True


def get_various_speed(l: list):
    n = len(l)
    l.sort()
    print(f'平均值,中位数,90%百分位,95%百分位,99%百分位,最小值,最大值: ', np.mean(l), l[int(n*0.5)], l[int(n*0.9)], l[int(n*0.95)], l[int(n*0.99)], min(l), max(l))
    keys_l = '平均值,中位数,90%百分位,95%百分位,99%百分位,最小值,最大值'.split(',')
    ori_values_l = [np.mean(l), l[int(n*0.5)], l[int(n*0.9)], l[int(n*0.95)], l[int(n*0.99)], min(l), max(l)]
    values_d = {}
    for key, value in zip(keys_l, ori_values_l):
        values_d[key] = [int(value*1000)]
    values = pd.DataFrame(values_d)
    with pd.ExcelWriter('./speed.xlsx') as w:
        values.to_excel(w, index=False)



def judge_except_pattern(name: str, except_pattern: Dict[str, Union[str, list]]):    
    """if one pattern is matched, return True.
    The key of except patterns is: 'start', 'mid', 'end'.
    """    
    def build_pattern(flag: str, pattern: str) -> str:
        assert flag in flag_l, "flag is illegal!"
        if flag == 'start':
            return '^' + re.escape(pattern) + '.+'
        elif flag == 'end':
            return '.+' + re.escape(pattern) + '$'
        else:
            return '.*' + re.escape(pattern) + '.*'
        
    def judge_pattern(flag, except_pattern):
        if isinstance(except_pattern, str):
            if re.match(build_pattern(flag, except_pattern), name):
                return True
        else:
            for start_pattern_elem in except_pattern:
                if re.match(build_pattern(flag, except_pattern), name):
                    return True
    
    flag_l = ['start', 'mid', 'end']
    for flag in flag_l:
        if flag in except_pattern and judge_pattern(flag, except_pattern[flag]):
            return True
        

def get_file_paths_r(file_dir: str, except_pattern: Union[List, str]=None) -> list:
    """get all file paths recursively.
    Args:
        file_dir: the dir including files.
    Returns:
        origin_path_list: a list including all file paths.
    """
    origin_path_list = []
    for root, _, files in os.walk(file_dir, topdown=False):
        for name in files:
            if judge_except_pattern(name, except_pattern):
                continue
            file_path = os.path.join(root, name)
            origin_path_list.append(file_path)
    return origin_path_list


def get_dir_tree_1level(src_dir: str) -> Dict[str, list]:
    """The Function can only process 1 level recursion.
    """
    res_d = defaultdict(list)
    names_l = os.listdir(src_dir)
    for name in names_l:
        sub_dir_path = osp.join(src_dir, name)
        res_d[name] = set(os.listdir(sub_dir_path))
    return res_d


def writeDataIntoExcel(xlsxPath: str, data: dict): 
	"""xlsxPath must be xlsx file.
	"""	
	writer = pd.ExcelWriter(xlsxPath)
	sheetNames = data.keys() # 获取所有sheet的名称
	# sheets是要写入的excel工作簿名称列表
	data = pd.DataFrame(data)
	for sheetName in sheetNames:
		data.to_excel(writer, sheet_name=sheetName)
	# 保存writer中的数据至excel
	# 如果省略该语句，则数据不会写入到上边创建的excel文件中
	writer.save() 
        

def writeDataIntoCSV(csvPath: str, data: dict): 
	"""xlsxPath must be csv file.
	"""	
	data_pd = pd.DataFrame(data)
	data_pd.to_csv(csvPath)
        

def get_dirs_numbers(dst_dir: str, names_s: Union[set, dict, None]=None) -> None:
    """get the file number of every dir
    """
    res_d = {}
    for root, dirs, files in os.walk(dst_dir):
        for dir_ in dirs:
            if names_s and dir_ not in names_s:
                continue
            res_d[dir_] = get_files_number(osp.join(root, dir_))
    return res_d


def mkdir_dir_tree(main_path: str, dir_tree: Union[Dict[str, List[str]], str], exist_rm: bool=False) -> None:
    """Create dirs recursively.

    Args:
        main_path: where to do creating dirs.
        dir_tree: a dir tree to create.

    Usage Example:
        dir_path = '/data'
        name2idx = eval(utils_.load_txt('./documents_classify/data_process/name2idx.txt'))
        dir_tree_d = {'raw': list(name2idx.keys())}
        utils_.mkdir_dir_tree(dir_path, dir_tree_d)
    """
    if isinstance(dir_tree, str): 
        new_dir_path = osp.join(main_path, dir_tree)
        mkdir_dir(new_dir_path, exist_rm)
        return
    for key in dir_tree:
        sub_main_path = osp.join(main_path, key)
        mkdir_dir(sub_main_path, exist_rm)
        for sub_dir_tree in dir_tree[key]:
            mkdir_dir_tree(sub_main_path, sub_dir_tree)


def get_dir_tree(dir_path: str) -> Union[dict, str]:
    """get the tree structure of the destination dir recursively.
    """
    def helper(main_path: str, name: str):
        dir_path = ospj(main_path, name)
        if osp.isfile(dir_path): return name
        res_d = defaultdict(list)
        for sub_name in os.listdir(dir_path):
            if judge_except_pattern(sub_name, {'start': '.'}):
                continue
            res_d[name].append(helper(dir_path, sub_name))
        return res_d
    main_dir_path, name = osp.dirname(dir_path), osp.basename(dir_path)
    return helper(main_dir_path, name)

        
def parse_dir_tree(dir_path: str, tree: dict):
    """parse the dir tree got by get_dir_tree.

    Args: 
        dir_path: the main dir path.
        tree: the dir tree in dir_path.
    """
    if isinstance(tree, str): 
        yield ospj(dir_path, tree)
        return 
    for name in tree:
        sub_dir_path = ospj(dir_path, name)
        for sub_tree in tree[name]:
            for i in parse_dir_tree(sub_dir_path, sub_tree):
                yield i


def sync_files2remote(src_dir: str, remote_dst_dir: str, ip: str, port: str, rsa_path: str) -> None:
    """upload file into the remote server, after which delete uploaded local file.
    """
    while True:
        process_n = fore_process_n = 0
        tree_d = get_dir_tree(src_dir) # read the dir tree in advance to prevent loading broken file.
        scr_super_dir = osp.dirname(src_dir)
        for file_path in parse_dir_tree(scr_super_dir, tree_d): # load each file path
            print(file_path)
            remote_sub_dir = file_path.replace(src_dir, remote_dst_dir)
            os.system(
                f"scp  -i {rsa_path} -P {port} {file_path} root@{ip}:{remote_sub_dir}"
            )
            os.remove(file_path)
            process_n += 1
        if process_n == 0:
            time.sleep(5)
        elif fore_process_n == 0:
            return 
        fore_process_n = process_n


def upload_file(remote_dir: str, file_path: str, ip: str, port: str, rsa_path: str) -> None:
    os.system(
                f"scp  -i {rsa_path} -P {port} {file_path} root@{ip}:{remote_dir}"
            )


def save_pkl(pkl_path: str, content: Any) -> None:
    with open(pkl_path,"wb") as f:
        dill.dump(content, f)


def load_pkl(pkl_path: str) -> Any:
    with open(pkl_path,"rb") as f:
        res = dill.load(f)
    return res
    

def rotate_clockwise_90(img):
    img = cv2.transpose(img)
    return cv2.flip(img, 1)

def rotate_clockwise_180(img):
    img = cv2.flip(img, 0)
    return cv2.flip(img, 1)

def rotate_clockwise_270(img):
    img = cv2.transpose(img)
    return cv2.flip(img, 0)

def rotate_fix(img, class_id):
    if class_id == 1:
        return rotate_clockwise_270(img)
    elif class_id == 2:
        return rotate_clockwise_180(img)
    elif class_id == 3:
        return rotate_clockwise_90(img)
    else:
        return img
    



def listCreatTree(root, llist, i): 
    if i < len(llist):
        if llist[i] == None:
            return None  ###这里的return很重要
        else:
            root = TreeNode(llist[i])
            # 往左递推
            root.left = listCreatTree(root.left, llist, 2 * i + 1)  # 从根开始一直到最左，直至为空，
            # 往右回溯
            root.right = listCreatTree(root.right, llist, 2 * i + 2)  # 再返回上一个根，回溯右，
            # 再返回根'
            return root  ###这里的return很重要
    return root


# 层次遍历: 利用队列
from collections import deque
def level_order(root):
    queue = deque()
    queue.append(root)
    while queue:
        node = queue.popleft()
        print(node.val, end=',')
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)


# 去掉行号
def remove_line(s):
    # s = """ 1 dd = {'banana': 3, 'apple':4, 'pear': 1, 'orange': 2}
    # 2 #按key排序
    # 3 kd = collections.OrderedDict(sorted(dd.items(), key=lambda t: t[0]))
    # 4 print kd
    # 5 #按照value排序
    # 6 vd = collections.OrderedDict(sorted(dd.items(),key=lambda t:t[1]))
    # 7 print vd
    # 8 
    # 9 #输出
    # 10 OrderedDict([('apple', 4), ('banana', 3), ('orange', 2), ('pear', 1)])
    # 11 OrderedDict([('pear', 1), ('orange', 2), ('banana', 3), ('apple', 4)])"""
    l = s.split('\n')
    for idx, elem in enumerate(l):
        elem = elem.split(' ')
        for idx1, elem1 in enumerate(elem):
            if elem1.isdigit():
                # print('elem1: ', elem1)
                l[idx] = ' '.join(elem[idx1+1:])
                # print(l[idx])
                break
    return '\n'.join(l)


def change_list(l_s):
    l_s = l_s.strip('[').strip(']')
    res = l_s.split(',')
    for idx, i in enumerate(res):
        if i == 'null':
           res[idx] = None
        else:
            res[idx] = int(i)
    return res


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def get_real_code(s):
    """
    rm ">>" in code: 

    s = ">> a=Variable(torch.Tensor([1]),requires_grad=False) 
        >> b=Variable(torch.Tensor([2]),requires_grad=True)
        >> c=a+b
        >> c.backward()
        >> a.grad  # 因为a的requires_grad=False 所以不存储梯度值
        >> b.grad"
    """
    l = s.split('\n')
    patterns_l = ['>> ', '>>>']
    for idx, elem in enumerate(l):
        for patttern in patterns_l:
            if patttern in elem and elem.startswith(patttern):
                l[idx] = elem.replace(patttern, '')
                break
    print('\n'.join(l))
    return '\n'.join(l)


class TimeoutException(Exception):
    def __init__(self, error='Timeout waiting for response from Cloud'):
        Exception.__init__(self, error)


def timeout_limit(timeout_time):
    """
    It's a decorator. 
    raise eroror if the timeout occurs for func
    """
    def wraps(func):
        def handler(signum, frame):
            raise TimeoutException()

        def deco(*args, **kwargs):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout_time)
            func(*args, **kwargs)
            signal.alarm(0)
        return deco
    return wraps


def get_doc_size(path):
    try:
        size = os.path.getsize(path)
        return get_mb_size(size)
    except Exception as err:
        print(err)

def get_mb_size(bytes):
    bytes = float(bytes)
    mb = bytes / 1024 / 1024
    return mb


def compress_images(dest_path, im):
    cv2.imwrite(dest_path, im, [cv2.IMWRITE_JPEG_QUALITY, 10]) 


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time() 
        func(*args, **kwargs)
        end_time = time.time()
        print(f'running time: {end_time - start_time}')
    return wrapper


def get_md5(file_path):
    md5 = None
    if os.path.isfile(file_path):
        f = open(file_path,'rb')
        md5_obj = hashlib.md5()
        md5_obj.update(f.read())
        hash_code = md5_obj.hexdigest()
        f.close()
        md5 = str(hash_code).lower()
    return md5