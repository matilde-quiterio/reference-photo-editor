import image_analysis as anal
import image_io as funcs

path_tgt = "data/target/target.jpg"
path_ref = "data/reference/reference.jpg"

image_target = funcs.load_image(path_tgt)
image_ref = funcs.load_image(path_ref)

brightness, contrast, saturation, temperature = anal.estimate_basic_adjustments(image_target,image_ref)
edited_image = funcs.apply_basic_edit(image_target, brightness, contrast, saturation, temperature)

funcs.save_image(edited_image, "data/output/edit.jpg")
