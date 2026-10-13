from PIL import Image
import numpy as np

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

image = "data/target/target.jpg"
image = load_image(image)
image_contrast_05 = adjust_contrast(image,0.5)
image_contrast_1 = adjust_contrast(image,1)
image_contrast_15 = adjust_contrast(image,1.5)


save_image(image_contrast_05, "data/output/image_contrast_05.jpg")
save_image(image_contrast_1, "data/output/image_contrast_1.jpg")
save_image(image_contrast_15, "data/output/image_contrast_15.jpg")
