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

def estimate_basic_adjustments(target_image, reference_image):
    brightness = analyze_brightness(reference_image) - analyze_brightness(target_image)
    contrast = analyze_contrast(reference_image) / analyze_contrast(target_image)
    saturation = analyze_saturation(reference_image) / analyze_saturation(target_image)

    return brightness,contrast,saturation

path_tgt = "data/target/target.jpg"
path_ref = "data/reference/reference.jpg"
image_target = funcs.load_image(path_tgt)
image_ref = funcs.load_image(path_ref)
brightness, contrast, saturation = estimate_basic_adjustments(image_target,image_ref)
funcs.apply_basic_edit(image_target, brightness, contrast, saturation, 0)