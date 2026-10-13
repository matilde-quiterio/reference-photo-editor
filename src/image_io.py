from PIL import Image
import numpy as np
import cv2


def load_image(path):
    pil_image = Image.open(path)  

    pil_image = pil_image.convert(mode="RGB")
    image_array = np.asarray(pil_image)

    return image_array

def save_image(image_array,path):
    pil_image = Image.fromarray(image_array)
    pil_image.save(path)

def adjust_brightness(image_array,value):
    image_float = image_array.astype(np.float32)
    image_float = image_float + value
    image_float = image_float.clip(0,255)
    image_int = image_float.astype(np.uint8)

    return image_int

def adjust_contrast(image_array, factor):
    image_float = image_array.astype(np.float32)
    image_float = (image_float - 128) * factor + 128 
    image_float = image_float.clip(0,255)
    image_int = image_float.astype(np.uint8)

    return image_int

def adjust_saturation(image_array, factor):
    image_hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    image_float = image_hsv.astype(np.float32)
    image_float[:,:,1] = image_float[:,:,1] * factor
    image_float = image_float.clip(0,255)
    image_int = image_float.astype(np.uint8)
    image_rgb = cv2.cvtColor(image_int, cv2.COLOR_HSV2RGB)

    return image_rgb

def adjust_temperature(image_array, value):
    image_float = image_array.astype(np.float32)
    image_float[:,:,0] = image_float[:,:,0] + value
    image_float[:,:,2] = image_float[:,:,2] - value
    image_float = image_float.clip(0,255)
    image_int = image_float.astype(np.uint8)

    return image_int

image = "data/target/target.jpg"
image = load_image(image)
output_imageneg = adjust_temperature(image, -30)
output_image = adjust_temperature(image, 0)
output_imagepos = adjust_temperature(image, 30)

save_image(output_imageneg, "data/output/temperature-.jpg")
save_image(output_image, "data/output/temperature.jpg")
save_image(output_imagepos, "data/output/temperature+.jpg")


print(image.shape)
print(output_image.shape)
print(image[0,0])
print(output_image[0,0])
