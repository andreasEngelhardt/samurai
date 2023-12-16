# Copyright 2022 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import os
from collections import namedtuple
from typing import List, Optional, Tuple, Union
import random
import json

import imageio
import numpy as np
from dataflow.samurai.quadrants import combine_direction, Direction

# imageio.plugins.freeimage.download()

DatasetReturn = namedtuple(
    "DatasetReturn",
    (
        "img_idx",
        "image_list",
    ),
)

def np_normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = np_normalize(z)
    vec1_avg = up
    vec0 = np_normalize(np.cross(vec1_avg, vec2))
    vec1 = np_normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = np_normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w


def recenter_poses(poses):
    # From https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/load_llff.py#L243

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


def recenter_poses_pts(poses, pts):
    '''normalize the poses based on their positions and viewing directions'''
    def poses_avg(poses, is360=False):
        hwf = poses[0, :3, -1:]
        center = min_line_dist(poses[:, :3, 3:4], poses[:, :3, 2:3])
        center = poses[:, :3, 3].mean(0)
        z = poses[:, :3, 2].sum(0)
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([viewmatrix(z, up, center, is360=is360), hwf], 1)
        return c2w

    hwf = poses[:,:3,4:] 
    c2w = poses_avg(poses, True)[:3,:4]
    
    # Multiply poses with c2w
    pose_dtype = poses.dtype
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w_homo = np.concatenate([c2w, bottom], -2)
    c2w_inv = np.linalg.inv(c2w_homo)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)
    poses = c2w_inv @ poses
    poses = np.concatenate([poses[:,:3,:4], hwf], 2).astype(pose_dtype)

    if pts is not None:
        pts = np.concatenate([pts, pts[:,0:1]*0+1], axis=-1)[:,:,np.newaxis]
        pts = (c2w_inv @ pts)[:, :3,0]    

    return poses, pts, c2w


def blender2opengl(transform: np.ndarray) -> np.ndarray:
    """Convert transform matrix from blender to opengl coordinate system.

        Blender coordinate system is assumed to be: x-right, y-back, z-up
        and opengl to be x-right, y-up, z-forward.
    """
    # Switch y and z axis and negate z coordinate.
    gl_to_blender = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    # Build a transform that converts to the blender coordinate system,
    # applies the original transform and then converts back to opengl.
    # return np.transpose(gl_to_blender) @ transform @ gl_to_blender
    if len(transform.shape) == 3:
        return gl_to_blender[np.newaxis] @ transform

    return  gl_to_blender @ transform


def colmap2opengl(transform: np.ndarray) -> np.ndarray:
    """Convert transform matrix from blender to opengl coordinate system.
    
        Blender coordinate system is assumed to be: x-right, y-back, z-up
        and opengl to be x-right, y-up, z-forward.    
    """
    # Switch y and z axis and negate z coordinate.
    gl_to_blender = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    # Build a transform that converts to the blender coordinate system, 
    # applies the original transform and then converts back to opengl.
    # return np.transpose(gl_to_blender) @ transform @ gl_to_blender
    if len(transform.shape) == 3:
        return gl_to_blender[np.newaxis] @ transform
    else:
        return  gl_to_blender @ transform


class SamuraiDataset:
    def __init__(self, data_dir: str, dirs_ext_to_read: List[Tuple[str, str]], scale:float = 1.0, load_gt_poses: bool = False, gt_poses_format: str = "blender"):
        super().__init__()
        data_dir = data_dir
        all_dirs = [os.path.join(data_dir, d[0]) for d in dirs_ext_to_read]
        img_names = sorted(
            [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(all_dirs[0]) if any([ext in f.lower() for ext in [".jpg", '.jpeg', ".png"]])],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )
        self.load_gt_poses = load_gt_poses

        if self.load_gt_poses:
            poses = []

            if "colmap" in gt_poses_format:
                poses_arr = np.load(os.path.join(data_dir, 'poses_bounds.npy'))
                # I think we need to go from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
                # This transformation pipeline is based on the original NeRF pipeline.
                # https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/load_llff.py#L243
                poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
                poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
                poses = np.moveaxis(poses, -1, 0).astype(np.float32)
                if not gt_poses_format == "colmap_orb":
                    poses = recenter_poses(poses)
                # Assuming a fixed focal length here which is the case for all colmap reconstructions.
                hwf = poses[:, :3, -1]
                # focal length in px to angle_x. 
                # print("hwf load", hwf)
                self.focal = 2 * np.arctan(hwf[:, 0:1] / (2 * hwf[:, 2:3]))
                # print("self.focal", self.focal)
                poses = poses[:, :3, :4]
                poses[:, :3, -1] *= scale
                if gt_poses_format == "colmap_neroic":
                    poses[:, 1, -1] += 0.5
                # Make homogeneous.
                h_poses = np.expand_dims(np.eye(4), 0)
                h_poses = np.tile(h_poses, (poses.shape[0], 1, 1))
                h_poses[:, :3, :] = poses
                poses = colmap2opengl(h_poses).astype(np.float32)
                # poses = h_poses.astype(np.float32)

                # if gt_poses_format == "colmap_neroic":
                #     poses = np.array(
                #         [
                #             [0, 0, -1, 0],
                #             [0, 1, 0, 0],
                #             [1, 0, 0, 0],
                #             [0, 0, 0, 1],
                #         ]
                #     ) @ poses
                # Check if we need to resort the poses.
                image_names_file = os.path.join(data_dir, 'image_names.pkl')
                if os.path.isfile(image_names_file):
                    # Get image order from this one.
                    print("Found image names file. Loading.")
                    imgfiles = pickle.load(open(image_names_file, "rb"))
                    imgnames = [int(os.path.splitext(imgf)[0]) for imgf in imgfiles]
                    ordered_idxs = np.argsort(imgnames)
                    poses = poses[ordered_idxs]
                    self.focal = self.focal[ordered_idxs]

                # if os.path.exists(os.path.join(data_dir, 'pts.pkl')):
                #     pts_arr = pickle.load(open(os.path.join(data_dir, 'pts.pkl'), "rb"))
                # if not gt_poses_format == "colmap_orb":
                #     poses = recenter_poses(poses)
                
                self.poses = poses
                print("Loaded gt poses: ", len(self.poses))
                self.directions = self.poses[:, :, 2]

            else:
                with open(os.path.join(data_dir, "quadrants.json"), "r") as fp:
                    metas = json.load(fp)
                    self.focal = metas["camera_angle_x"]
                for frame in metas["frames"]:
                    transform = np.array(frame["transform_matrix"])
                    transform[:, -1] *= scale
                    # Convert from blender to OpenGL coordinate system.
                    # print("Transform originaL:", transform)
                    transform = blender2opengl(transform)
                    # print("Transform converted:", transform)
                    poses.append(transform)
                self.poses = np.array(poses).astype(np.float32)
                self.directions = self.poses[:, :, 2]
            
            # if noise_on_gt_poses > 0:
            #     distance = np.mean(np.linalg.norm(self.poses[:, :, -1], axis=-1))
            #     noise_t = np.random.normal(
            #         size=(self.poses.shape[0], 3, 1), scale=(distance / 5) * noise_on_gt_poses
            #     )
            #     noise_r = np.random.normal(
            #         size=(self.poses.shape[0], 3, 3), scale=(np.pi * 0.5) * (noise_on_gt_poses / 5)
            #     )

            #     self.poses[:, :3, -2:-1] += noise_t
            #     self.poses[:, :3, :3] += noise_r
            #     # print(self.poses.shape)
        else:
            with open(os.path.join(data_dir, "quadrants.json"), "r") as fp:
                self.quadrant_info = json.load(fp)

            self.directions = np.stack(
                [
                    combine_direction(*[Direction(e) for e in d["quadrant"]])
                    for d in self.quadrant_info["frames"]
                ],
                0,
            ).astype(np.float32)
            print(self.directions.shape)

        extensions = [d[1] for d in dirs_ext_to_read]

        self.channels = [d[2] for d in dirs_ext_to_read]

        self.num_img_types = len(dirs_ext_to_read)
        self.image_type_paths = [
            [os.path.join(p, n + e) for n in img_names]  # Path/Name.Extension
            for p, e in zip(
                all_dirs, extensions
            )  # all dirs and extensions are derived from dirs_ext_to_read
        ]

    def get_poses(self):
        return self.poses
    
    def get_focal(self, max_size):
        if self.load_gt_poses:
            focal = []
            img_type = self.image_type_paths[0]
            for i in range(self.__len__()):
                dims = np.asarray(self.read_image(img_type[i]).shape[:-1])
                max_dim = np.max(dims)
                scaler = max_size / max_dim
                w = (dims[1]).astype(np.float32)
                if isinstance(self.focal, np.ndarray) and len(self.focal) > 1:
                    f = self.focal[i]
                else:
                    f = self.focal
                focal.append((0.5 * w / np.tan(0.5 * f)) * scaler)
            return np.reshape(np.asarray(focal).astype(np.float32), [-1, 1])
        else:
            return None
    
    def get_directions(self):
        return self.directions

    def __len__(self):
        return len(self.image_type_paths[0])

    def get_image_shapes(self, max_size):
        img_type = self.image_type_paths[0]
        shapes = []
        for i in range(self.__len__()):
            dims = np.asarray(self.read_image(img_type[i]).shape[:-1])
            max_dim = np.max(dims)
            factor = max_size / max_dim
            dims = (dims * factor).astype(np.int32)

            shapes.append(dims)

        return np.stack(shapes, 0)

    def file_ids_to_indices(self, ids):
        # Convert image name based ids to indices.
        def img_type_name_to_int(t, i):
            number = os.path.splitext(os.path.split(t[i])[1])[0]
            if number.isnumeric():
                return int(number)
            return i

        # Assuming rgb and mask images have the same naming.
        img_type = self.image_type_paths[0]
        indices_list = [img_type_name_to_int(img_type, i) for i in range(len(img_type))]
        indices_list = np.asarray(indices_list, dtype=np.int32)
        print("DEBUG: indices list", indices_list)

        indices = []
        for i in range(len(ids)):
            indices.append(np.where(indices_list==ids[i])[0][0])
        return np.asarray(indices, dtype=np.int32)

    def read_image(self, path) -> np.ndarray:
        img = imageio.imread(path)
        if img.dtype == np.uint8:
            img = img / 255
        img = np.nan_to_num(img)
        return img

    def get_image_iter(self, shuffle=False):
        idxs = list(range(len(self.image_type_paths[0])))

        if shuffle:
            random.shuffle(idxs)

        for idx in idxs:
            yield self[idx]

    def get_image_paths(self):
        idxs = list(range(len(self.image_type_paths[0])))
        imgs = []

        for img_type in self.image_type_paths:
            types = []
            for idx in idxs:
                types.append(img_type[idx])

            imgs.append(types)

        return (idxs, *imgs)

    def get_image_path_from_indices(self, idxs):
        if isinstance(idxs, list) or isinstance(idxs, np.ndarray):
            imgs = []
            for img_type in self.image_type_paths:
                types = []

                for idx in idxs:
                    types.append(img_type[idx])
                imgs.append(types)

            return (idxs, *imgs)
        else:
            import tensorflow as tf

            if isinstance(idxs, tf.Tensor):
                types = []
                for img_type in self.image_type_paths:
                    types.append(tf.gather_nd(img_type, tf.reshape(idxs, (-1, 1))))

                imgs = tf.stack(types, 0)

                type_list = tf.unstack(imgs, num=len(self.image_type_paths))
                return (idxs, *type_list)

    def __getitem__(self, idx):
        return (
            idx,
            np.stack(
                [self.read_image(img_type[idx]) for img_type in self.image_type_paths],
                0,
            ),
        )
