# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:11:33 2020

@author: Javier
"""
import numpy as np
import os, csv, natsort
from skimage import io
import skimage.color
from rle import rle_encode
from project1 import skin_lesion_segmentation
from project1 import evaluate_masks
from test_csv_for_kaggle import test_prediction_csv
import copy
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt



#CARGAMOS LAS IMÁGENES Y MÁSCARAS DE TRAIN
data_dir= ''

train_imgs_files = [os.path.join(data_dir,'train/images',f) for f in sorted(os.listdir(os.path.join(data_dir,'train/images'))) 
            if (os.path.isfile(os.path.join(data_dir,'train/images',f)) and f.endswith('.jpg'))]

train_masks_files = [os.path.join(data_dir,'train/masks',f) for f in sorted(os.listdir(os.path.join(data_dir,'train/masks'))) 
            if (os.path.isfile(os.path.join(data_dir,'train/masks',f)) and f.endswith('.png'))]

#Ordenamos para que cada imagen se corresponda con cada máscara
train_imgs_files.sort()
train_masks_files.sort()
print("Número de imágenes de train", len(train_imgs_files))
print("Número de máscaras de train", len(train_masks_files))

#CARGAMOS LAS IMÁGENES DE TEST
test_imgs_files = [os.path.join(data_dir,'test/images',f) for f in sorted(os.listdir(os.path.join(data_dir,'test/images'))) 
            if (os.path.isfile(os.path.join(data_dir,'test/images',f)) and f.endswith('.jpg'))]

test_imgs_files.sort()
print("Número de imágenes de test", len(test_imgs_files))



#1.Con el algoritmo creado en "skin_lesion_segmentation", 
#comprobamos qué tal es nuestra nota de segmentación mediante 
img_roots = train_imgs_files.copy()
gt_masks_roots = train_masks_files.copy()


mean_score = evaluate_masks(img_roots, gt_masks_roots)


#Una vez satisfechos con el resultado, generamos el fichero para hacer la submission en Kaggle
dir_images_name = 'test/images'
csv_name='test_prediction_rgb2g_radialmask_otsu_closing_fill_holes.csv'
test_prediction_csv(dir_images_name, csv_name)