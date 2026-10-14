import image_analysis as anal
import image_io as funcs

path_tgt = "data/target/target.jpg"
path_ref = "data/reference/ref3.jpg"

image_target = funcs.load_image(path_tgt)
image_ref = funcs.load_image(path_ref)

brightness, contrast, saturation, temperature = anal.estimate_basic_adjustments(image_target,image_ref)
edited_image = funcs.apply_basic_edit(image_target, brightness, contrast, saturation, temperature)
lab_image = anal.color_transfer_lab(image_target, image_ref)
blend_image25 = anal.hybrid_edit(image_target, lab_image, 0.25)
blend_image50 = anal.hybrid_edit(image_target, lab_image, 0.5)
blend_image75 = anal.hybrid_edit(image_target, lab_image, 0.75)

funcs.save_image(edited_image, "data/output/edit.jpg")
funcs.save_image(lab_image, "data/output/lab.jpg")
funcs.save_image(blend_image25, "data/output/hybrid25.jpg")
funcs.save_image(blend_image50, "data/output/hybrid50.jpg")
funcs.save_image(blend_image75, "data/output/hybrid75.jpg")
