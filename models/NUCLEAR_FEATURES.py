import copy
import json
import time

import math
import os
import random
import re
import uuid
from io import BytesIO
from xml.etree import ElementTree as ET
import cv2
import pandas
import torch
from skimage import measure, feature
from scipy import ndimage as ndi

import numpy
import numpy as np

import pandas as pd
import pyclipper
from Crypto.Cipher import AES
from concavity import gaussian_smooth_geom
from matplotlib import pyplot
from scipy.spatial import distance
from shapely.geometry import Polygon
from skimage import measure
from skimage import morphology


def nuclear_features(original_image, segmentation_image):
    '''

    Haoda Lu, Computational Digital Pathology Lab (CDPL), Bioinformatics Institute, A*STAR, Singapore. Email:
    lu_haoda@bii.a-star.edu.sg
    2024/06/21


    Multi nuclear features for machine learning classifer construction.

    包括形态学特征（如主轴长度、次轴长度等）、纹理特征和拓扑结构特征。我们将使用以下特征：

    形态学特征：
    面积（Area）
    周长（Perimeter）
    长轴长度（Major Axis Length）
    短轴长度（Minor Axis Length）
    离心率（Eccentricity）
    密实度（Solidity）
    形状因子（Shape Factor = 4πArea / Perimeter²）
    凸面积（Convex Area）
    等效直径（wozheduan ）

    纹理特征：
    对比度（Contrast）
    不相似性（Dissimilarity）
    同质性（Homogeneity）
    二阶矩（ASM）
    能量（Energy）
    相关性（Correlation）

    拓扑结构特征：
    欧拉数（Euler Number）
    '''
    if original_image.shape[0] < 10 or original_image.shape[1] < 10:
        return ''

    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    # 确保分割图像是二值图像
    _, binary_image = cv2.threshold(segmentation_image, 128, 255, cv2.THRESH_BINARY)

    # 标记连接区域
    labeled_image, num_labels = ndi.label(binary_image)

    # 获取不同类别的标签
    unique_labels = np.unique(labeled_image)
    unique_labels = unique_labels[unique_labels != 0]  # 移除背景标签

    # 初始化存储特征的字典
    features_by_category = {}

    # 遍历每个类别
    for label in unique_labels:
        # 获取当前类别的二值掩码
        category_mask = (labeled_image == label).astype(np.uint8)

        # 计算形态学特征
        props = measure.regionprops(category_mask)
        morph_features = []
        for prop in props:
            morph_features.append({
                'Area': prop.area,
                'Perimeter': prop.perimeter,
                'Major Axis Length': prop.major_axis_length,
                'Minor Axis Length': prop.minor_axis_length,
                'Eccentricity': prop.eccentricity,
                'Solidity': prop.solidity,
                'Shape Factor': (4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter != 0 else 0,
                'Convex Area': prop.convex_area,
                'Equivalent Diameter': prop.equivalent_diameter
                })

        # 计算纹理特征（使用灰度共生矩阵）
        glcm = feature.greycomatrix(original_image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = feature.greycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = feature.greycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = feature.greycoprops(glcm, 'homogeneity')[0, 0]
        ASM = feature.greycoprops(glcm, 'ASM')[0, 0]
        energy = feature.greycoprops(glcm, 'energy')[0, 0]
        correlation = feature.greycoprops(glcm, 'correlation')[0, 0]

        texture_features = {
            'Contrast': contrast,
            'Dissimilarity': dissimilarity,
            'Homogeneity': homogeneity,
            'ASM': ASM,
            'Energy': energy,
            'Correlation': correlation
            }

        # 计算拓扑结构特征（例如欧拉数）
        regionprops_table = measure.regionprops_table(category_mask, properties=['label', 'euler_number'])
        euler_number = regionprops_table['euler_number'][0] if 'euler_number' in regionprops_table else None

        # 存储当前类别的特征
        features_by_category[str(label)] = {
            'Morphological Features': morph_features,
            'Texture Features': texture_features,
            'Topological Features': {'Euler Number': euler_number}
            }

    # # 输出特征
    for label, features in features_by_category.items():

        # print(f"类别 {label} 的特征:")
        # print("形态学特征:")
        features['Morphological Features'] = list(features['Morphological Features'])
        for i, morph_feature in enumerate(features['Morphological Features']):
            for key, value in features['Morphological Features'][i].items():
                features['Morphological Features'][i][key] = float(value)
            # print(f"细胞核 {i + 1}: {morph_feature}")

        # print("\n纹理特征:")
        # print(features['Texture Features'])
        for key, value in features['Texture Features'].items():
            features['Texture Features'][key] = float(value)

        # print("\n拓扑结构特征:")
        # print(features['Topological Features'])
        for key, value in features['Topological Features'].items():
            features['Topological Features'][key] = float(value)
        # print("\n" + "-" * 40 + "\n")

        features_by_category[label] = features

    features_by_category = json.dumps(features_by_category)
    # print(features_by_category)
    return features_by_category