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

def get_lab_stats(image_array):
    img_lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    l_mean = img_lab[:,:,0].mean()
    a_mean = img_lab[:,:,1].mean()
    b_mean = img_lab[:,:,2].mean()

    l_std = img_lab[:,:,0].std()
    a_std = img_lab[:,:,1].std()
    b_std = img_lab[:,:,2].std()

    return l_mean, a_mean,b_mean, l_std, a_std, b_std

def color_transfer_lab(target_image, reference_image):
    target_lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2LAB)
    l_mean_target,a_mean_target,b_mean_target, l_std_target,a_std_target,b_std_target = get_lab_stats(target_image)
    l_mean_ref,a_mean_ref,b_mean_ref,l_std_ref,a_std_ref,b_std_ref = get_lab_stats(reference_image)
    target_l = target_lab[:,:,0].astype(np.float32)
    target_a = target_lab[:,:,1].astype(np.float32)
    target_b = target_lab[:,:,2].astype(np.float32)

    l_new = (target_l - l_mean_target) /l_std_target * l_std_ref + l_mean_ref
    a_new = (target_a - a_mean_target) /a_std_target * a_std_ref + a_mean_ref
    b_new = (target_b - b_mean_target) /b_std_target * b_std_ref + b_mean_ref

    l_new = l_new.clip( 0, 255).astype(np.uint8)
    a_new = a_new.clip( 0, 255).astype(np.uint8)
    b_new = b_new.clip( 0, 255).astype(np.uint8)


    target_lab[:,:,0] = l_new
    target_lab[:,:,1] = a_new
    target_lab[:,:,2] = b_new

    target_rgb = cv2.cvtColor(target_lab, cv2.COLOR_LAB2RGB)

    return target_rgb
