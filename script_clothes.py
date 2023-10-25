import rembg

def remove_background(input_path, output_path):
    with open(input_path, "rb") as image_file:
        with open(output_path, "wb") as output_file:
            img_data = image_file.read()
            result = rembg.remove(img_data)
            output_file.write(result)