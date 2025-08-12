import time
import argparse
import os

import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--front_image", type=str, required=True)
parser.add_argument("--left_image", type=str, required=True)
parser.add_argument("--back_image", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

images = {
    "front": args.front_image,
    "left": args.left_image,
    "back": args.back_image
}

for key in images:
    image = Image.open(images[key]).convert("RGBA")
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)
    images[key] = image

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mv',
    subfolder='hunyuan3d-dit-v2-mv',
    variant='fp16'
)

start_time = time.time()
mesh = pipeline(
    image=images,
    num_inference_steps=50,
    octree_resolution=380,
    num_chunks=20000,
    generator=torch.manual_seed(12345),
    output_type='trimesh'
)[0]
print("--- taking %s seconds ---" % (time.time() - start_time))
print(f'Saving mesh to {args.output_dir}/mesh.obj')
os.makedirs(args.output_dir, exist_ok=True)
mesh.export(f'{args.output_dir}/mesh.obj')
