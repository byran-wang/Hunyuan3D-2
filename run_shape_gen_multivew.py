import os

# scenes = ["7ab52011f82f60e5d06b5582cd874bf6a7b0288489fe3827aec0829dfee71399",
#           "f5548e475296967ad9797bc51fa23738648c2def23082dce6fd300c271439b69",
#           "4f4616dc1b2be598e4332130aee5be906b4ea7ce81d8f6b8cda1e59bd1ae1526",
#           "03554445ec87ee6dc62ae789015c51fa7a4f4f6802f3b87bc9ce090775620854",
#           "bc8d0eb8f1525ee9ab39b00a197980f14ffce1b7d83531373e62696f28c18852",
# ]
input_folder = "/home/simba/Documents/project/TRELLIS/datasets/ABO/renders_cond"
scenes = os.listdir(input_folder)
out_base = "outputs_HOI3D_multi"


for scene in scenes:
    # list all images in the scene folder
    images = os.listdir(os.path.join(input_folder, scene))
    # sort images by name
    images.sort()
    input_idx = [7, 14, 21]
    input1 = os.path.join(input_folder, scene, images[input_idx[0]])
    input2 = os.path.join(input_folder, scene, images[input_idx[1]])
    input3 = os.path.join(input_folder, scene, images[input_idx[2]])
    sub_folder = "_".join([images[i].split(".")[0] for i in input_idx])
    output_dir = os.path.join(out_base, scene, sub_folder)
    cmd = f"python examples/shape_gen_multiview.py --front_image {input1} --left_image {input2} --back_image {input3} --output_dir {output_dir}"
    print(cmd)
    os.system(cmd)
