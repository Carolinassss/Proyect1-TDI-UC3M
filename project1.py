#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %      
% TDImagen PROYECTO 1: SEGMENTACIÓN DE IMÁGENES                         %
%                                                                       %
% Plantilla para implementar la función principal del sistema,          %
% 'skin_lesion_segmentation', que recibe como entrada la ruta a una     %
% imagen de una lesión y, a su salida, proporciona una máscara de       %
% segmentación, predicha a partir de la solución propuesta.             %
%                                                                       %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import numpy as np
from skimage import io, filters, color, morphology, draw
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
from scipy import ndimage

def skin_lesion_segmentation(img_root):
    """ SKIN_LESION_SEGMENTATION: ... 
    - - -  COMPLETAR - - - 
    """
    # El siguiente código implementa el BASELINE incluido en el challenge de
    # Kaggle. 
    # - - - MODIFICAR PARA IMPLEMENTACIÓN DE LA SOLUCIÓN PROPUESTA. - - -
    image = io.imread(img_root)
    image_gray = color.rgb2gray(image)  
    M,N = image_gray.shape
    
    # PREPROCESADO
    image_gray = filters.gaussian(image_gray,3)
    center = (int(M/2), int(N/2))
    # radius = min(center[0], center[1], M-center[0], N-center[1])
    # Y, X = np.ogrid[:M, :N]
    # dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    # mask = dist_from_center <= radius
    # Create the basic mask
    mask = np.ones(shape=image_gray.shape[0:2], dtype="bool")
    # Draw filled rectangle on the mask image
    rr, cc = draw.ellipse(center[0],center[1],center[0],center[1])
    mask[rr, cc] = 0
    mask = np.invert(mask)

    
    
    
    # SEGMENTACION
    otsu_th = filters.threshold_otsu(image_gray)
    predicted_mask = (image_gray < otsu_th).astype('int')
    
    predicted_mask = predicted_mask*mask
    
    # POSTPROCESADO
    predicted_mask = morphology.binary_closing(predicted_mask)
    predicted_mask = ndimage.binary_fill_holes(predicted_mask)
    predicted_mask = ndimage.binary_fill_holes(predicted_mask)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    return predicted_mask
        
def evaluate_masks(img_roots, gt_masks_roots):
    """ EVALUATE_MASKS: Función que, dadas dos listas, una con las rutas
        a las imágenes a evaluar y otra con sus máscaras Ground-Truth (GT)
        correspondientes, determina el Mean Average Precision a diferentes
        umbrales de Intersección sobre Unión (IoU) para todo el conjunto de
        imágenes.
    """
    score = []
    for i in np.arange(np.size(img_roots)):
        predicted_mask = skin_lesion_segmentation(img_roots[i])
        gt_mask = io.imread(gt_masks_roots[i])/255 
        image = io.imread(img_roots[i])/255 
        score.append(jaccard_score(np.ndarray.flatten(gt_mask),np.ndarray.flatten(predicted_mask)))
        
        if i==1:
            plt.figure()
            plt.subplot(2,3,1)
            plt.imshow(image,'gray')
            plt.subplot(2,3,2)
            plt.imshow(predicted_mask,'gray')
            plt.subplot(2,3,3)
            plt.imshow(gt_mask,'gray')
        if i==2:
            plt.subplot(2,3,4)
            plt.imshow(image,'gray')
            plt.subplot(2,3,5)
            plt.imshow(predicted_mask,'gray')
            plt.subplot(2,3,6)
            plt.imshow(gt_mask,'gray')
            
            
    mean_score = np.mean(score)
    print('Jaccard Score sobre el conjunto de imágenes proporcionado: '+str(mean_score))
    
    return mean_score
