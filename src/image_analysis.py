import numpy as np
import cv2

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
    temperature = analyze_temperature(reference_image) - analyze_temperature(target_image)

    brightness = np.clip(brightness, -30,30)
    contrast = np.clip(contrast, 0.8, 1.2)
    saturation = np.clip(saturation, 0.8, 1.3)
    temperature = np.clip(temperature, -30,30)

    return brightness,contrast,saturation, temperature

def analyze_temperature(image_array):
    red_mean = image_array[:,:,0].mean()
    blue_mean = image_array[:,:,2].mean()

    temp_score = red_mean - blue_mean
    return temp_score

