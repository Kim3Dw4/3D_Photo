# Repo source: https://github.com/vt-vl-lab/3d-photo-inpainting

#import os
#os.environ['QT_DEBUG_PLUGINS'] = '1'

import subprocess
#subprocess.run('ldd /home/user/.local/lib/python3.8/site-packages/PyQt5/Qt/plugins/platforms/libqxcb.so', shell=True)
#subprocess.run('pip list', shell=True)
subprocess.run('nvidia-smi', shell=True)

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1920, 1080)).start()
#subprocess.run('echo $DISPLAY', shell=True)

# 3d inpainting imports
import numpy as np
import argparse
import glob
import os
from functools import partial
import vispy
import scipy.misc as misc
from tqdm import tqdm
import yaml
import time
import sys
from mesh import write_ply, read_ply, output_3d_photo
from utils import get_MiDaS_samples, read_MiDaS_depth
import torch
import cv2
from skimage.transform import resize
import imageio
import copy
from networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net
from MiDaS.run import run_depth
from boostmonodepth_utils import run_boostmonodepth
from MiDaS.monodepth_net import MonoDepthNet
import MiDaS.MiDaS_utils as MiDaS_utils
from bilateral_filtering import sparse_bilateral_filtering

import torch

# gradio imports
import gradio as gr
import uuid
from PIL import Image
from pathlib import Path
import shutil
from time import sleep

def inpaint(img_name, num_frames, fps):
    
    config = yaml.load(open('argument.yml', 'r'))
    
    config['num_frames'] = num_frames
    config['fps'] = fps
    
    if torch.cuda.is_available():
        config['gpu_ids'] = 0
    
    if config['offscreen_rendering'] is True:
        vispy.use(app='egl')
    
    os.makedirs(config['mesh_folder'], exist_ok=True)
    os.makedirs(config['video_folder'], exist_ok=True)
    os.makedirs(config['depth_folder'], exist_ok=True)
    sample_list = get_MiDaS_samples(config['src_folder'], config['depth_folder'], config, config['specific'], img_name.stem)
    normal_canvas, all_canvas = None, None

    if isinstance(config["gpu_ids"], int) and (config["gpu_ids"] >= 0):
        device = config["gpu_ids"]
    else:
        device = "cpu"

    print(f"running on device {device}")

    for idx in tqdm(range(len(sample_list))):
        depth = None
        sample = sample_list[idx]
        print("Current Source ==> ", sample['src_pair_name'])
        mesh_fi = os.path.join(config['mesh_folder'], sample['src_pair_name'] +'.ply')
        image = imageio.imread(sample['ref_img_fi'])

        print(f"Running depth extraction at {time.time()}")
        if config['use_boostmonodepth'] is True:
            run_boostmonodepth(sample['ref_img_fi'], config['src_folder'], config['depth_folder'])
        elif config['require_midas'] is True:
            run_depth([sample['ref_img_fi']], config['src_folder'], config['depth_folder'],
                      config['MiDaS_model_ckpt'], MonoDepthNet, MiDaS_utils, target_w=640)

        if 'npy' in config['depth_format']:
            config['output_h'], config['output_w'] = np.load(sample['depth_fi']).shape[:2]
        else:
            config['output_h'], config['output_w'] = imageio.imread(sample['depth_fi']).shape[:2]
        frac = config['longer_side_len'] / max(config['output_h'], config['output_w'])
        config['output_h'], config['output_w'] = int(config['output_h'] * frac), int(config['output_w'] * frac)
        config['original_h'], config['original_w'] = config['output_h'], config['output_w']
        if image.ndim == 2:
            image = image[..., None].repeat(3, -1)
        if np.sum(np.abs(image[..., 0] - image[..., 1])) == 0 and np.sum(np.abs(image[..., 1] - image[..., 2])) == 0:
            config['gray_image'] = True
        else:
            config['gray_image'] = False
        image = cv2.resize(image, (config['output_w'], config['output_h']), interpolation=cv2.INTER_AREA)
        depth = read_MiDaS_depth(sample['depth_fi'], 3.0, config['output_h'], config['output_w'])
        mean_loc_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
        if not(config['load_ply'] is True and os.path.exists(mesh_fi)):
            vis_photos, vis_depths = sparse_bilateral_filtering(depth.copy(), image.copy(), config, num_iter=config['sparse_iter'], spdb=False)
            depth = vis_depths[-1]
            model = None
            torch.cuda.empty_cache()
            print("Start Running 3D_Photo ...")
            print(f"Loading edge model at {time.time()}")
            depth_edge_model = Inpaint_Edge_Net(init_weights=True)
            depth_edge_weight = torch.load(config['depth_edge_model_ckpt'],
                                           map_location=torch.device(device))
            depth_edge_model.load_state_dict(depth_edge_weight)
            depth_edge_model = depth_edge_model.to(device)
            depth_edge_model.eval()

            print(f"Loading depth model at {time.time()}")
            depth_feat_model = Inpaint_Depth_Net()
            depth_feat_weight = torch.load(config['depth_feat_model_ckpt'],
                                           map_location=torch.device(device))
            depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
            depth_feat_model = depth_feat_model.to(device)
            depth_feat_model.eval()
            depth_feat_model = depth_feat_model.to(device)
            print(f"Loading rgb model at {time.time()}")
            rgb_model = Inpaint_Color_Net()
            rgb_feat_weight = torch.load(config['rgb_feat_model_ckpt'],
                                         map_location=torch.device(device))
            rgb_model.load_state_dict(rgb_feat_weight)
            rgb_model.eval()
            rgb_model = rgb_model.to(device)
            graph = None


            print(f"Writing depth ply (and basically doing everything) at {time.time()}")
            rt_info = write_ply(image,
                                  depth,
                                  sample['int_mtx'],
                                  mesh_fi,
                                  config,
                                  rgb_model,
                                  depth_edge_model,
                                  depth_edge_model,
                                  depth_feat_model)

            if rt_info is False:
                continue
            rgb_model = None
            color_feat_model = None
            depth_edge_model = None
            depth_feat_model = None
            torch.cuda.empty_cache()
        if config['save_ply'] is True or config['load_ply'] is True:
            verts, colors, faces, Height, Width, hFov, vFov = read_ply(mesh_fi)
        else:
            verts, colors, faces, Height, Width, hFov, vFov = rt_info


        print(f"Making video at {time.time()}")
        videos_poses, video_basename = copy.deepcopy(sample['tgts_poses']), sample['tgt_name']
        top = (config.get('original_h') // 2 - sample['int_mtx'][1, 2] * config['output_h'])
        left = (config.get('original_w') // 2 - sample['int_mtx'][0, 2] * config['output_w'])
        down, right = top + config['output_h'], left + config['output_w']
        border = [int(xx) for xx in [top, down, left, right]]
        normal_canvas, all_canvas = output_3d_photo(verts.copy(), colors.copy(), faces.copy(), copy.deepcopy(Height), copy.deepcopy(Width), copy.deepcopy(hFov), copy.deepcopy(vFov),
                            copy.deepcopy(sample['tgt_pose']), sample['video_postfix'], copy.deepcopy(sample['ref_pose']), copy.deepcopy(config['video_folder']),
                            image.copy(), copy.deepcopy(sample['int_mtx']), config, image,
                            videos_poses, video_basename, config.get('original_h'), config.get('original_w'), border=border, depth=depth, normal_canvas=normal_canvas, all_canvas=all_canvas,
                            mean_loc_depth=mean_loc_depth)

def resizer(input_img, max_img_size=512):
    width, height = input_img.size
    long_edge = height if height >= width else width
    if long_edge > max_img_size:
        ratio = max_img_size / long_edge
        resized_width = int(ratio * width)
        resized_height = int(ratio * height)
        resized_input_img = input_img.resize((resized_width, resized_height), resample=2)
        return resized_input_img 
        
    else:
        return input_img

def main_app(input_img, num_frames, fps):
    
    # resize down
    input_img = resizer(input_img)
    
    # Save image in necessary folder for inpainting
    #img_name = Path(str(uuid.uuid4()) + '.jpg')
    img_name = Path('sample.jpg')
    save_folder = Path('image')
    input_img.save(save_folder/img_name)
    
    inpaint(img_name, num_frames, fps)
    
    #subprocess.run('ls -l', shell=True)
    #subprocess.run('ls image -l', shell=True)
    #subprocess.run('ls video/ -l', shell=True)
    
    # Get output video path & return
    input_img_path = str(save_folder/img_name)
    out_vid_path = 'video/{0}_circle.mp4'.format(img_name.stem)
    
    return out_vid_path

video_choices = ['dolly-zoom-in', 'zoom-in', 'circle', 'swing']
gradio_inputs = [gr.inputs.Image(type='pil', label='Input Image'),
                 gr.inputs.Slider(minimum=60, maximum=240, step=1, default=120, label="Number of Frames"),
                 gr.inputs.Slider(minimum=10, maximum=40, step=1, default=20, label="Frames per Second (FPS)")]
                 
gradio_outputs = [gr.outputs.Video(label='Output Video')]
examples = [ ['moon.jpg'], ['dog.jpg'] ]

description="Convert an image into a trajectory-following video. Images are automatically resized down to a max edge of 512. | NOTE: The current runtime for a sample is around 400-700 seconds. Running on a lower number of frames could help! Do be patient as this is on CPU-only, BUT if this space maybe gets a GPU one day, it's already configured to run with GPU-support :) If you have a GPU, feel free to use the author's original repo (linked at the bottom of this path, they have a collab notebook!) You can also run this space/gradio app locally!"

article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2004.04727' target='_blank'>3D Photography using Context-aware Layered Depth Inpainting</a> | <a href='https://shihmengli.github.io/3D-Photo-Inpainting/' target='_blank'>Github Project Page</a> | <a href='https://github.com/vt-vl-lab/3d-photo-inpainting' target='_blank'>Github Repo</a></p>"

iface = gr.Interface(fn=main_app, inputs=gradio_inputs , outputs=gradio_outputs, examples=examples,
                     title='3D Image Inpainting',
                     description=description,
                     article=article,
                     enable_queue=True)

iface.launch(debug=True)