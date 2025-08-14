import time

import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--image_file', type=str, default='')
parser.add_argument('--output_dir', type=str, default='outputs')
args = parser.parse_args()

image_path = args.image_file
image = Image.open(image_path)

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2',
    subfolder='hunyuan3d-dit-v2-0',
    variant='fp16'
)

start_time = time.time()
mesh = pipeline(image=image,
                num_inference_steps=50,
                octree_resolution=380,
                num_chunks=20000,
                generator=torch.manual_seed(12345),
                output_type='trimesh',
                output_dir=args.output_dir,
                )[0]
print("--- %s seconds ---" % (time.time() - start_time))
mesh.export(os.path.join(args.output_dir, 'mesh.obj'))
