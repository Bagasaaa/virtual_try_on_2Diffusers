from tqdm import tqdm
import os

from cloths_segmentation.pre_trained_models import create_model
import argparse
import torch
import cv2
import numpy as np

from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
import albumentations as albu

# create model
model = create_model("Unet_2020-10-30")
model.eval()

def get_cloth_mask(input, output, name):
  image = load_rgb(r"{}".format(input))

  transform = albu.Compose([albu.Normalize(p=1)], p=1)

  padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)

  x = transform(image=padded_image)["image"]

  x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

  with torch.no_grad():
    prediction = model(x)[0][0]

  mask = (prediction > 0).cpu().numpy().astype(np.uint8)
  mask = unpad(mask, pads)

  cv2.imwrite(r"{}".format(output) + "\{}.jpg".format(name), cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * 255)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='input output folder for openpose')

    parser.add_argument('--dir_image', type=str, required=True)
    parser.add_argument('--out_image', type=str, required=True)
    
    args = parser.parse_args()

    index = 1
    for i in tqdm(os.listdir(fr"{args.dir_image}")):
        get_cloth_mask(fr"{args.dir_image}" + "\\" + fr"{i}", fr"{args.out_image}", index)
        index += 1
