import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import kiui
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

from PIL import Image
import json

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ABODataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training

        if self.training:
            self.image_dir = '/data2/yumi/abo-benchmark-material/train_data'
        else:
            self.image_dir = '/data2/yumi/abo-benchmark-material/test_data'
        # self.image_dir = '/data2/yumi/abo-benchmark-material/train_data'
        self.pose_dir = '/data2/yumi/abo-benchmark-material/cam_pose'
        self.mask_dir = '/data2/yumi/abo-benchmark-material/abo-benchmark-material_all'
        self.items = os.listdir(self.image_dir)

        # # naive split
        # if self.training:
        #     self.items = self.items[:-self.opt.batch_size]
        # else:
        #     self.items = self.items[-self.opt.batch_size:]
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        uid = self.items[idx]
        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0

        # TODO: choose views, based on your rendering settings
        if self.training:
            # input views are in (36, 72), other views are randomly selected
            vids = np.random.permutation(np.arange(36, 73))[:self.opt.num_input_views].tolist() + np.random.permutation(90).tolist()
        else:
            # fixed views
            vids = np.arange(36, 73, 4).tolist() + np.arange(90).tolist()
        
        for vid in vids:

            image_path = os.path.join(self.image_dir, uid, f'seg_{vid}.jpg')
            camera_path = os.path.join(self.pose_dir, uid, 'metadata.json')
            mask_path = os.path.join(self.mask_dir, uid, f'segmentation/segmentation_{vid}.jpg')
            # camera_path = os.path.join(pose_dir, uid, f'{vid:03d}.txt')

            try:
                # TODO: load data (modify self.client here)
                # image = np.frombuffer(self.client.get(image_path), np.uint8)
                # image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                # c2w = [float(t) for t in self.client.get(camera_path).decode().strip().split(' ')]
                # c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)

                # image = Image.open(image_path).convert('RGBA') 
                with open(image_path, 'rb') as f:
                    binary_data = f.read()
                image_np = np.frombuffer(binary_data, dtype=np.uint8)
                image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)
                # image = np.array(image).astype(np.float32) / 255.0
                image = image.astype(np.float32) / 255.0 
                image = torch.from_numpy(image)
                
                with open(camera_path, 'r') as file:
                    pose_data = json.load(file)
                c2w = pose_data['views'][vid]['pose']
                c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4,4)

            except Exception as e:
                print(f'[WARN] dataset {uid} {vid}: {e}')
                continue
            
            # TODO: you may have a different camera system
            # blender world + opencv cam --> opengl world & cam
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1 # invert up and forward direction

            # scale up radius to fully use the [-1, 1]^3 space!
            c2w[:3, 3] *= self.opt.cam_radius / 1.5 # 1.5 is the default scale
          
            image = image.permute(2, 0, 1) # [4, 512, 512]
            # mask = image[3:4] # [1, 512, 512]
            mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.0 
            mask = torch.from_numpy(mask).unsqueeze(0)

            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]
        assert len(images) == len(masks) == len(cam_poses), f'images:{len(images)}, masks:{len(masks)},cam_poses:{len(cam_poses)}'

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # data augmentation
        if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

     
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        return results