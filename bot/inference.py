from imports import add_path
add_path()
import generator as gen
import config as config
from config import load_weights_inference, DEVICE
from dataset import  transform, denorm
from PIL import Image
from torchvision.utils import save_image
import time

def prepare_generator():
    generator = gen.Generator().to(config.DEVICE).eval()
    load_weights_inference(config.WEIGHTS_GEN_B,generator)

    return generator

def inference(image_name, model):
    image = Image.open(image_name).convert('RGB')
    image_name = image_name.split(".")[0]
    image = transform(image)
    image = image.to(DEVICE)
    # save_image(denorm(image), f"{image_name}_.png")
    fake_image = model(image)
    saved_path = f"{image_name}__.png"
    save_image(denorm(fake_image), saved_path)
    
    return saved_path

if __name__ == "__main__":
    start_prepairing = time.time()
    gena = prepare_generator()
    end_prepairing = time.time()
    delta = end_prepairing - start_prepairing
    print(f"prepairing took: {delta}")

    # ([1, 3, 256, 256])
    # ([3, 256, 256])
    # ([1, 3, 256, 256])