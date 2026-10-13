from PIL import Image
import numpy as np
import cv2
import image_io as funcs

def analyze_brightness(image_array):
    img_hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    img_hsv_mean = img_hsv[:,:,2].mean()
    return img_hsv_mean

def analyze_contrast(image_array):
    img_hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    img_hsv_std = img_hsv[:,:,2].std()
    return img_hsv_std

def analyze_saturation(image_array):
    img_hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    img_hsv_mean = img_hsv[:,:,1].mean()
    return img_hsv_mean
path_tgt = "data/target/target.jpg"
path_ref = "data/reference/reference.jpg"
image_target = funcs.load_image(path_tgt)
image_ref = funcs.load_image(path_ref)
print(f'Brightness target: ', analyze_brightness(image_target), 'Ref:', analyze_brightness(image_ref))
print(f'Contrast: ', analyze_contrast(image_target),'Ref:', analyze_contrast(image_ref))
print(f'Saturation: ', analyze_saturation(image_target),'Ref:', analyze_saturation(image_ref))
