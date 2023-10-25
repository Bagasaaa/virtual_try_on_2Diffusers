import cv2
import os
from flask import Flask, request, render_template
from PIL import Image
import subprocess
from script_diffusers import pipe, image_grid

import insightface
from insightface.app import FaceAnalysis

import torch

from diffusers import StableDiffusionInpaintPipeline

from PIL import Image

from PIL import Image
from io import BytesIO

import gdown
import json

app = Flask(__name__, template_folder='templates')

app_face_analysis = FaceAnalysis(name='buffalo_l', root=".")
app_face_analysis.prepare(ctx_id=0, det_size=(640, 640))

# if not os.path.exists("models/swapper.onnx"):
#             gdown.download(
#                 "https://drive.google.com/uc?export=download&id=1imzdF1O6YIcCmKykAQE_T7qu_550dArC",
#                 output="models/inswapper_128.onnx"
#             )

swapper = insightface.model_zoo.get_model('./models/inswapper_128.onnx')

FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN = (255, 255, 0)

app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}

def allowed_photo(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")

@app.route('/change_clothes', methods=['GET', 'POST'])
def change_clothes():
    if request.method == 'POST':
        user_selfie_photo = request.files.get("user_selfie_photo")
        user_height = int(request.form.get("user_height"))
        user_weight = int(request.form.get("user_weight"))
        user_race = request.form.get("user_race").lower()
        prompt = str(request.form.get('prompt')).lower()
        negative_prompt = str(request.form.get('negative_prompt')).lower()
        user_choose_outfit_what_to_change = str(request.form.get("user_choose_outfit_what_to_change")).lower()

        output_image_dir = "./output_image_dir/"
        masked_image_dir = "./output/alpha"

        selfie_filepath = os.path.join(output_image_dir, "selfie.jpg")
        user_selfie_photo.save(selfie_filepath)

        # Load the data from the JSON file
        with open('photo_category.json', 'r') as file:
            data = json.load(file)

        # Check if user_height_weight_race exists in the JSON data and return the appropriate category
        selected_category = None
        for category, category_data in data.items():
            height_range = category_data["user_height"]
            weight_range = category_data["user_weight"]
            user_races = category_data["user_race"]
            # print(category_data)
            if height_range["min"] <= user_height <= height_range["max"] and weight_range["min"] <= user_weight <= weight_range["max"]:  # Menggunakan any() untuk pengecekan
                selected_category = category
                # print(selected_category)
                break
        # print(selected_category)
        if not selected_category:
            return "Kategori tidak dikenali"

        # Load the user image based on the selected category
        image_path = os.path.join("./pose_foto", user_race, selected_category, "image.jpg")
        print(image_path)

        user_image = Image.open(image_path).resize((512, 512))

        # Simpan gambar user_image sebagai file sementara
        user_image_path = "./user_image/user_image.jpg"
        user_image.save(user_image_path)

        if user_choose_outfit_what_to_change == "top":
            mask_filename = "1"
        elif user_choose_outfit_what_to_change == "bottom":
            mask_filename = "2"
        else:
            return "No change detected"

        # Call the get_cloth_mask function for the selected outfit
        # get_cloth_mask(user_image_path, masked_image_dir, mask_filename)
        subprocess.run(f"python huggingface-cloth-segmentation/process.py --image {user_image_path}", shell=True)

        # Convert the user_selfie_photo and mask file to Image objects
        mask_image_path = os.path.join(masked_image_dir, mask_filename + ".png")
        mask_image = Image.open(mask_image_path).resize((512, 512))

        # Perform the diffusion process
        guidance_scale = 7.5
        num_samples = 1
        generator = torch.Generator(device="cuda").manual_seed(0)  # Change the seed to get different results

        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=user_image,
            mask_image=mask_image,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_samples,
        ).images

        images.insert(0, user_image)

        resized_images = []
        for image in images:
            # Resize the image
            resized_image = image.resize((720, 1280))

            # Append the resized image to the list
            resized_images.append(resized_image)

        grid_image = image_grid(resized_images, 1, num_samples+1)

        grid_filepath = os.path.join(output_image_dir, "grid.jpg")
        grid_image.save(grid_filepath)

        img1 = cv2.imread(r".\output_image_dir\grid.jpg")
        print(img1)
        img2 = cv2.imread(r".\output_image_dir\selfie.jpg")
        print(img2)
        face1 = app_face_analysis.get(img1)
        face2 = app_face_analysis.get(img2)

        res = swapper.get(img1, face1[0], face2[0], paste_back=True)

        # res = res[:, :, ::-1]
        # Save the grid_image to the output directory
        # grid_filepath = os.path.join(output_image_dir, "grid.jpg")
        cv2.imwrite("a.jpg", res)

        return "Change Clothes process completed."

    return render_template('change_clothes.html')

if __name__ == '__main__':
    app.run(debug=True)