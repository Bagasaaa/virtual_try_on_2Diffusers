# virtual_try_on_2Diffusers

Virtual Try On 2D using Inswapper and Inpainting Model. Users can customize their body weight and height.
Many thanks to this repository https://github.com/wildoctopus/huggingface-cloth-segmentation
I use it for cloth segmentation (users can choose whether to change the upper or bottom).

I collaborate to do this project with a nice teammate [silvering-steve](https://github.com/silvering-steve/)

I use insightface, so you can replace the face in an existing photo, simply with a clear selfie photo of the desired face.

I am using the stable-diffusion-inpainting model from diffusers to swap clothes, but don't worry as you don't need to perform manual inpainting.

This is the example result. I just chose to change the upper clothes.
![a](https://github.com/Bagasaaa/virtual_try_on_2Diffusers/assets/119937815/c4c6f831-7054-499f-ac19-a0588db02ec3)

## How to Use?
Just hit "python app.py"
