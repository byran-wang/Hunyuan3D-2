import time

import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, default='datasets/real_multiview')
parser.add_argument('--output_base', type=str, default='outputs_real_multiview_masked')
args = parser.parse_args()

# list all images in input_folder
scenes = [f for f in os.listdir(args.input_folder) if os.path.isdir(os.path.join(args.input_folder, f))]

for scene in scenes:
    images = [f for f in os.listdir(os.path.join(args.input_folder, scene)) if f.endswith('.png')]
    for image in images:
        image_path = os.path.join(args.input_folder, scene, image)
        cmd = ""
        sub_folder = image_path.split('/')[-1].split('.')[0]
        output_dir = os.path.join(args.output_base, scene, sub_folder)
        os.makedirs(output_dir, exist_ok=True)
        cmd += f"python -m examples.shape_gen --image_file {image_path} --output_dir {output_dir}"
        subprocess.run(cmd, shell=True)
