import os
import cv2
import random
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
import kiui
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

#D:\anaconda\envs\lgm2\lib\site-packages\xformers\__init__.py 에서 triton 주석처리함.
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

        # TODO: remove this barrier
       # self._warn()

        # TODO: load the list of objects for training
        # self.items = []
        # with open('D:/dataset/object_list.txt', 'r') as f: #jpg, txt있는 경로를 합친 파일의 경로
        #     for line in f.readlines():
        #         self.items.append(line.strip())


        # naive split
        # batch_size만큼을 test 항목으로 사용
        # if self.training:
        #     self.items = self.items[:-self.opt.batch_size]
        # else:
        #     self.items = self.items[-self.opt.batch_size:]
        

        # default camera intrinsics
        # 카메라 시야각 fovy를 기반으로 행렬 계산 / 3D -> 2D
        """
        >> opt <<
        fovy: float = 49.1
        znear: float = 0.5  / 근거리 비율
        zfar: float = 2.5   / 원거리 비율
        """
        # 초기화
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))

        # x & y 축 scaling
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov

        # depth 반영
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)

        # 원근 투영에서 z 좌표를 w 좌표로 변환 / 3D -> 2D
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
            # multi-view 이미지들 중 특정 부분을 선택
            # 여기서는 36 - 73까지 무작위로 train / 36 -73까지 4간격으로 test
            vids = np.random.permutation(np.arange(36, 73))[:self.opt.num_input_views].tolist() + np.random.permutation(91).tolist()
        else:
            # fixed views
            vids = np.arange(36, 73, 4).tolist() + np.arange(91).tolist()
        

        for vid in vids:
            # image_path = os.path.join('D:/dataset/spins/seg_sample/B07B4MF6YH/',f"{uid}.jpg")#, 'rgb', f'{vid:03d}.jpg')
            # camera_path = os.path.join('D:/code/LGM-main/poses2/',f"{uid}.txt")#, 'pose', f'{vid:03d}.txt')
            # mask_path = os.path.join('D:/dataset/spins/segmentation',f"{uid}.jpg")
            image_path = os.path.join(self.image_dir, uid, f'seg_{vid}.jpg')
            camera_path = os.path.join(self.pose_dir, uid, 'metadata.json')
            mask_path = os.path.join(self.mask_dir, uid, f'segmentation/segmentation_{vid}.jpg')

            try:
                # TODO: load data (modify self.client here)
                #client.get함수는 원격으로 서버나 데이터베이스에서 가져오는것임.로컬 컴에서 불러올려면 client를 삭제하고 다른 함수를 써야함

                # 로드된 binary 데이터를 NumPy 배열로 변환 (데이터형 : uint8)
                #image = np.frombuffer(self.client.get(image_path), np.uint8)
                # NumPy 배열 --> 실제 이미지로 디코딩
                #image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                
                #image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
                #image = torch.from_numpy(image)
                with open(image_path, 'rb') as f:
                    binary_data = f.read()
                image_np = np.frombuffer(binary_data, dtype=np.uint8)
                image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)

                image = image.astype(np.float32) / 255
                image = torch.from_numpy(image)  

                mask = np.array(Image.open(mask_path)).astype(np.float32) / 255
                mask = torch.from_numpy(mask).unsqueeze(0)

                with open(camera_path, 'r') as file:
                    pose_data = json.load(file)
                c2w = pose_data['views'][vid]['pose']
                c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4,4)

                # camera pose 데이터 로드 --> 문자열로 디코딩 --> 공백으로 분리된 문자열 --> 실수 리스트
                # with open(camera_path, 'r') as f:
                #     c2w = [float(t) for t in f.read().strip().strip().split()]
                #     # tensor로 변환
                #     c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)

            except Exception as e:
                print(f'[WARN] dataset {uid} {vid}: {e}')
                continue
            # TODO: you may have a different camera system
            # blender world + opencv cam --> opengl world & cam
            # 사용하는 좌표계가 다르기 때문에 변환
            # y축 <--> z축 방향 변환 & 순서 변환
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1 # invert up and forward direction

            # scale up radius to fully use the [-1, 1]^3 space!
            c2w[:3, 3] *= self.opt.cam_radius / 1.5 # 1.5 is the default scale

            # PyTorch = [채널, 높이, 너비] 형식 / OpenCV = [높이, 너비, 채널] 형식
            # => permute로 채널 순서 변경          
            image = image.permute(2, 0, 1) # [4, 512, 512] 
            #mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break


        # view 수 부족할 때
        """
        >> opt <<
        num_views: int = 12
        cam_radius: float = 1.5 # to better use [-1, 1]^3 space
        num_input_views: int = 4
        input_size: int = 256       -> 512...?
        output_size: int = 256      -> 512...?
        """
        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            # 마지막 요소를 복제하여 부족한 수 채우기 (개인적으로 좋은 방법같지는 않음요...)
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          

        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        # 첫 번째 cam pose를 원점에서 self.opt.cam_radius만큼 z축 방향으로 이동
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        # transform 행렬을 4차원 텐서로 확장 -> 모든 cam_poses에 적용
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        # 입력으로 사용할 view 수 선택 및 복제
        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # data augmentation
        """
        >> utils <<
        grid_distortion     : strength 비율로 image에 grid 왜곡 적용
        orbit_camera_jitter : strength 비율로 jitter시켜 새로운 변환 pose 생성
        """
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
        """
        >> utils <<
        get_rays : 3D space에서 각 픽셀로부터 발사되는 rays의 방향과 원점을 반환
        """
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