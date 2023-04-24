import math
import os
from collections import namedtuple
import subprocess
from typing import List, Optional, Tuple, Union
import random
import json

import imageio
from nn_utils.math_utils import convert3x4_4x4
import numpy as np
from scipy import ndimage
from tqdm import tqdm
from dataflow.samurai.quadrants import combine_direction, Direction, direction_dict



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


def get_gl_intrinsic_matrix(
    focal_length_pixels: float, height: float, width: float):
  """Convert focal length of a pinhole camera to GL intrinsic matrix for rendering."""
  return np.array([
      [focal_length_pixels * 2 / width, 0, 0, 0],
      [0, focal_length_pixels * 2 / height, 0, 0],
      [0, 0, .0001, -0.002],
      [0, 0, 1.0, 0]]).astype(np.float32)


def apply_colors_to_depth_map(depth, minn=None, maxx=None):
  mask = (depth != 0.)
  if minn is None:
    minn = depth[mask].min()
  if maxx is None:
    maxx = depth[mask].max()
  norm = matplotlib.colors.Normalize(vmin=minn, vmax=maxx)
  mapper = cm.ScalarMappable(norm=norm, cmap='plasma')
  depth_colored = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
  depth_colored[~mask, :] = 0.
  return depth_colored


# def sample_points_from_mesh(mesh: tf.Tensor, num_sample_points: int):
#   """Samples points on a mesh, uniformly distributed over the surface area."""
#   surface_areas = (
#       tf.norm(
#           tf.linalg.cross(mesh[:, 0, :], mesh[:, 1, :])
#           + tf.linalg.cross(mesh[:, 1, :], mesh[:, 2, :])
#           + tf.linalg.cross(mesh[:, 2, :], mesh[:, 0, :]),
#           axis=-1) / 2)

#   cdf = tf.pad(tf.cumsum(surface_areas), [[1, 0]])
#   cdf /= cdf[-1]
#   rv = tf.random.uniform([num_sample_points])
#   triangle_index = tf.searchsorted(cdf, rv, side="right") - 1
#   assert (
#       tf.reduce_min(triangle_index) >= 0
#       and tf.reduce_max(triangle_index) < mesh.shape[0]
#   )

#   sampled_tri = tf.gather(mesh, triangle_index)
#   r1, r2 = tf.unstack(tf.random.uniform([num_sample_points, 2]), 2, -1)
#   r1 = tf.sqrt(r1)
#   u = 1 - r1
#   v = (1 - r2) * r1
#   w = r2 * r1
#   sampled_pts = (
#       sampled_tri[:, 0] * u[:, None]
#       + sampled_tri[:, 1] * v[:, None]
#       + sampled_tri[:, 2] * w[:, None])

#   return sampled_pts, triangle_index


# def get_valid_sample_coordinates(
#     triangles: tf.Tensor, sampled_points: tf.Tensor,
#     annotation, image: Image.Image):
#   """Get the sample coordinates that are visible from the particular image."""
#   translation = tf_transformations.translate(annotation['camera']['t'])
#   rotation = quaternion.qvec2rotmat(annotation['camera']['q'], expand=True)
#   object_to_world = translation @ rotation
#   h, w = annotation['image_size']
#   focal_length_pixels = annotation['camera']['focal_length']
#   intrinsics = get_gl_intrinsic_matrix(focal_length_pixels, h, w)
  
#   # Render the 3D model alignment
#   triangles_aligned = tf_transformations.transform_mesh(
#       triangles, object_to_world)
#   rend = scene_renderer.render_scene(
#         triangles_aligned, view_projection_matrix=intrinsics,
#         output_type=tf.float32, clear_color=(0,0,0,0),
#         image_size=image.size[::-1], cull_backfacing=False, return_rgb=False)
#   depth = rend[:, :, 3].numpy()

#   # Align the sampled points.
#   sampled_points_world = tf_transformations.transform_points(
#       sampled_points, object_to_world)
#   sampled_points_screen = tf_transformations.transform_points(
#       sampled_points_world, intrinsics)
#   sampled_points_screen += [1., 1., 0]
#   sampled_points_screen *= [image.size[0]/2, image.size[1]/2, 1]
#   samples = tf.concat(
#       (sampled_points_screen[:, :2], sampled_points_world[:, 2:3]),
#       axis=1).numpy()

#   # Discard points where the depth doesn't match the depth buffer.
#   coords = {}
#   for i_sample, sample in enumerate(samples):
#     y = round(sample[1])
#     x = round(sample[0])
#     z = sample[2]
#     if abs(depth[y, x] - z) < 1:
#       coords[i_sample] = (y, x)
#   return coords


def np_translate(translation):
    # Create a 4x4 matrix based on translation vector (3,).
    translation = np.asarray(translation).astype(np.float32)
    result = np.eye(4).astype(np.float32)
    result[:translation.size, -1] = translation
    return result


def np_qvec2rotmat(q, expand: bool = False):
    # Converts quaternion vector into rotation matrix. 
    # Input can be a list or array.
    rot_mat = np.array(
        [
            [
                1 - 2 * q[2]**2 - 2 * q[3]**2, 2 * q[1] * q[2] - 2 * q[0] * q[3],
                2 * q[3] * q[1] + 2 * q[0] * q[2]
            ],
            [
                2 * q[1] * q[2] + 2 * q[0] * q[3], 1 - 2 * q[1]**2 - 2 * q[3]**2,
                2 * q[2] * q[3] - 2 * q[0] * q[1]
            ],
            [
                2 * q[3] * q[1] - 2 * q[0] * q[2], 2 * q[2] * q[3] + 2 * q[0] * q[1],
                1 - 2 * q[1]**2 - 2 * q[2]**2
            ]
        ],
        dtype=np.float32
    )
    if expand:
        rot_mat = np.concatenate(
            [
                np.concatenate([rot_mat, np.zeros((1, rot_mat.shape[1]))], 0),
                np.concatenate([np.zeros((rot_mat.shape[1], 1)), [[1.]]], 0),
            ],
            1
        )
    return rot_mat


def camera_matrices_from_annotation(annotation):
  """Convert camera pose and intrinsics to 4x4 matrices."""
  translation = np_translate(annotation['camera']['t'])
  rotation = np_qvec2rotmat(annotation['camera']['q'], expand=True)
  object_to_world = translation @ rotation
  h, w = annotation['image_size']
  focal_length_pixels = annotation['camera']['focal_length']
  intrinsics = get_gl_intrinsic_matrix(focal_length_pixels, h, w)
  return (
    object_to_world.astype(np.float32),
    intrinsics.astype(np.float32),
    focal_length_pixels
)


def load_json_annotation(file_path) -> dict:
    """Load a annotations.json file from navi scene. 

    Args:
        file_path (_type_): _description_

    Returns:
        dict: _description_
    """
    with open(file_path, "r") as fo:
        annotations = json.load(fo)
    return annotations


def magnitude(x):
    return np.sqrt(x.dot(x))


def c2w_to_quadrants(poses: np.ndarray):
    quadrants_list = []

    directions = direction_dict()
    for pose in poses:
        origin = pose[:3, 3]
        origin_norm = np_normalize(origin)
        # Assume that the origin has the correct axis layout: x - right, y - top, z-front.
        # Calculate the distance to all quadrants
        magnitudes = np.stack(
            [magnitude(p - origin_norm) for p, _ in directions], 0
        )
        # Select the minimum distance
        sorted_idx = np.argsort(magnitudes)[0]

        # Get the corresponding quadrant definitions
        _, dirs = directions[sorted_idx]
        dirs = [str(d) for d in dirs]
        quadrants_list.append(dirs)
    
    return quadrants_list


def generate_masks(src_paths: List[str], dest_paths: List[str]) -> None:
    """Call submodule to perform image matting for the provided image paths.

    Args:
        src_paths (List[str]): _description_
        dest_paths (List[str]): _description_
    """
    import argparse
    import shutil
    import sys
    from PIL import Image, ExifTags, ImageOps
    
    from external_run.__main__ import run_method

    def resave(path):
        extension = os.path.splitext(path)[1]
        save_path = os.path.join(os.path.dirname(os.path.dirname(path)), "image", os.path.basename(path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            with Image.open(path) as image:
                ImageOps.exif_transpose(image).save(save_path)
        except (AttributeError, KeyError, IndexError):
            # cases: image don't have getexif
            shutil.copy(path, save_path)
        return save_path

    # Some libraries in python honor exif rotation other not.
    # Apply the rotation so all behave the same
    # This will create a copy of the original images.
    new_image_paths = []
    for fp in src_paths:
        new_image_paths.append(resave(fp))
    original_dirname = os.path.dirname(src_paths[0])
    
    # cmd = [
    #     f"python -m external_run",
    #     "--method", "u2net",
    #     "--dataset_path", os.path.dirname(os.path.dirname(src_paths[0])),
    # ]
    # subprocess.run(" ".join(cmd), stderr=sys.stderr, stdout=sys.stdout, shell=True)
    print("Running u2net.")
    basefolder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    method_args = ["--dataset_path", os.path.dirname(os.path.dirname(src_paths[0]))]
    run_method(basefolder, "u2net", method_args)
    # cmd = [
    #     f"python -m external_run",
    #     "--method", "FBA_Matting",
    #     "--dataset_path", os.path.dirname(os.path.dirname(src_paths[0])),
    # ]
    # subprocess.run(" ".join(cmd), stderr=sys.stderr, stdout=sys.stdout, shell=True)

    print("Running FBA_Matting.")
    run_method(basefolder, "FBA_Matting", method_args)

    print("Rename image folder. Backup original files.")
    os.rename(original_dirname, f"{original_dirname}_original")
    os.rename(os.path.dirname(new_image_paths[0]),  original_dirname)
    
    # Sanity check.
    for dp in dest_paths:
        if not os.path.isfile(dp):
            print("WARN: Sanity check failed. Could not find", dp)

def np_stable_invert_c2w(c2w):
    r = c2w[..., :3, :3]
    t = c2w[..., :3, 3]

    idxs = list(range(len(r.shape)))
    transpose_idxs = [*idxs[:-2], idxs[-1], idxs[-2]]

    r_inv = np.transpose(r, transpose_idxs)
    t_inv = np.sum(t[..., None, :] * r_inv, -1)

    w2c = np.concatenate([r_inv, -t_inv[..., None]], -1)
    w2c = convert3x4_4x4(w2c)

    return w2c


class NaviDataset:
    """Dataset for Navi dataset hosted inside google.
    """
    def __init__(self, scene_name: str,
                 data_root: str,
                 max_dimension_size: int,
                 preload: bool = True,
                 scene_type: str = "wild_set",
                 version: str = "v0.3",
                 load_gt_poses: bool = False, 
                 scale: float = 1.0, 
                 noise_on_gt_poses: float = 0.0,
                 skip_masks: bool = False,
                 automatic_scale: bool = False,
                 dirs_ext_to_read: Optional[List[Tuple[str, str]]] = None):
        super().__init__()
        
        self.navi_release_root = os.path.join(data_root, version)
        # Convert to Navi naming.
        object_id = scene_name
        scene_name = scene_type
        annotation_json_path = os.path.join(
            self.navi_release_root, object_id, scene_name, 'annotations.json'
        )
        annotations = load_json_annotation(annotation_json_path)

        self.load_gt_poses = load_gt_poses

        # Sort filepaths and extract image names.
        image_dir = os.path.join(
                self.navi_release_root, object_id, scene_name, 'images'
        )
        image_paths = []
        for i_anno, anno in enumerate(annotations):
            image_path = os.path.join(
            image_dir, anno['filename'])
            image_paths.append(image_path)
        img_names = sorted(
            [os.path.splitext(os.path.basename(f))[0] for f in image_paths],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )

        ext = os.path.splitext(image_paths[0])[1]
        self.filepaths = [os.path.join(image_dir, f"{n}{ext}") for n in img_names]
        mask_exts = [".png", ".jpg", ".exr"]
        mask_ext = ext
        for e in mask_exts:
            if os.path.isfile(os.path.join(self.navi_release_root, object_id, scene_name, "mask", f"{img_names[0]}{e}")):
                mask_ext = e
        self.mask_paths = [
            os.path.join(
                self.navi_release_root, object_id, scene_name, "mask", f"{n}{mask_ext}"
            ) for n in img_names
        ]
        if not os.path.isfile(self.mask_paths[0]) and not skip_masks:
            # Masks do probably not exist. Try to generate some.
            generate_masks(self.filepaths, self.mask_paths)

        self.max_dimension_size = max_dimension_size

        # Read all corresponding images to determine shapes.
        self.image_shapes = None
        self.original_image_shapes, self.image_shapes = self.get_image_shapes(max_dimension_size, get_orig_shapes=True)

        # Get the camera parameters.
        # Load poses from json.
        focal = []
        poses = []
        for i_anno, anno in enumerate(annotations):
            object_to_world, intrinsics, focal_length_pixel = camera_matrices_from_annotation(anno)
            cam_to_world = np_stable_invert_c2w(object_to_world[None])[0]
            # Go from colmap coordinate frame (right, down, fwd) to SAMURAI frame (right, up, fwd).
            cam_to_world = cam_to_world @ np.diag([1, -1, -1, 1])
            # For NeRF (right, up, back)
            # poses = poses @ np.diag([1, -1, -1, 1])
            poses.append(cam_to_world)
            focal.append(focal_length_pixel)

        self.poses = np.array(poses).astype(np.float32)
        if automatic_scale:
            # Determine scene scale based on heuristic.
            scene_radius = np.mean(np.linalg.norm(self.poses[..., :3, 3], axis=-1))
            scale = 1.0 / scene_radius
        self.poses[..., :3, 3] *= scale
        
        self.directions = self.poses[:, :, 2]
        self.focal = np.asarray(focal).astype(np.float32)

        if noise_on_gt_poses> 0:
            distance = np.mean(np.linalg.norm(self.poses[:, :, -1], axis=-1))
            noise_t = np.random.normal(
                size=(self.poses.shape[0], 3, 1), scale=(distance / 5) * noise_on_gt_poses
            )

            self.poses[:, :3, -2:-1] += noise_t
     
            noise_r = np.random.normal(
                size=(self.poses.shape[0], 3, 3), scale=(np.pi * 0.5) * (noise_on_gt_poses / 5)
            )
            self.poses[:, :3, :3] += noise_r

        if not load_gt_poses:
            # Convert poses to quadrant directions
            scene_dir = os.path.join(self.navi_release_root, object_id, scene_name)
            # Check if quadrants file exists.
            if os.path.isfile(os.path.join(scene_dir, "quadrants.json")):
                with open(os.path.join(scene_dir, "quadrants.json"), "r") as fp:
                    self.quadrant_info = json.load(fp)
                self.directions = np.stack(
                    [
                        combine_direction(*[Direction(e) for e in d["quadrant"]])
                        for d in self.quadrant_info["frames"]
                    ],
                    0,
                ).astype(np.float32)
            else:
                directions = c2w_to_quadrants(
                    self.poses
                )
                self.directions = np.stack(
                    [
                        combine_direction(*[Direction(e) for e in d])
                        for d in directions
                    ],
                    0,
                ).astype(np.float32)
                self.save_quadrants(os.path.join(scene_dir, "quadrants.json"), directions)
            self.poses = None

        if dirs_ext_to_read is not None:
            all_dirs = [os.path.join(data_dir, d[0]) for d in dirs_ext_to_read]
            self.channels = [d[2] for d in dirs_ext_to_read]
        else:
            all_dirs = [
                os.path.join(self.navi_release_root, object_id, scene_name, "images"),
                os.path.join(self.navi_release_root, object_id, scene_name, "mask")
            ]
            # Set channels for image and masks, respectively.
            self.channels = [3, 1]
        extensions = [os.path.splitext(d)[1] for d in [self.filepaths[0], self.mask_paths[0]]]

        self.image_type_paths = [
            [os.path.join(p, n + e) for n in img_names]  # Path/Name.Extension
            for p, e in zip(
                all_dirs, extensions
            )  # all dirs and extensions are derived from dirs_ext_to_read
        ]
        if preload:
            self.images = load_all_images()
        else:
            self.images = None


    def __len__(self):
        return len(self.filepaths)

    def save_quadrants(self, filepath, directions):
        
        frames = []
        for dirs in directions:
            frames.append(
                {
                    "quadrant": list(dirs)
                }
            )
        meta = {
            "frames": frames
        }
        quadrant_info = json.dumps(
            meta,
            indent=4,
        )

        with open(filepath, "w") as fp:
            fp.write(quadrant_info)
    
    def get_poses(self):
        return self.poses

    def get_directions(self):
        return self.directions

    def set_directions(self, indices):
        if self.directions is not None:
            self.directions = self.directions[indices]

    def set_poses(self, indices):
        if self.poses is not None:
            self.poses = self.poses[indices]

    def set_image_type_paths(self, indices):
        self.image_type_paths = [(np.array(type_paths)[indices]).tolist() for type_paths in self.image_type_paths]
    
    def get_focal(self, max_size):
        if self.load_gt_poses:
            format_scaler = self.image_shapes.max(axis=-1) / self.original_image_shapes.max(axis=-1)

            return np.expand_dims(self.focal * format_scaler, -1).astype(np.float32)
        else:
            return None

    def get_image_shapes(self, max_size, get_orig_shapes: bool = False):
        img_paths = self.filepaths
        if max_size == self.max_dimension_size and self.image_shapes is not None:
            if get_orig_shapes:
                return self.original_image_shapes, self.image_shapes
            return self.image_shapes
        shapes = []
        orig_shapes = []
        for i in tqdm(range(self.__len__()), desc="Reading image shapes"):
            dims = np.asarray(self.read_image(img_paths[i]).shape[:2])
            orig_shapes.append(dims)
            max_dim = np.max(dims)
            factor = max_size / max_dim
            dims = (dims * factor).astype(np.int32)

            shapes.append(dims)

        if get_orig_shapes:
            return np.stack(orig_shapes, 0), np.stack(shapes, 0)
        return np.stack(shapes, 0)
        

    def read_image(self, path) -> np.ndarray:
        img = imageio.imread(path)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255
        img = np.nan_to_num(img)
        return img

    def load_all_images(self):
        images = []
        
        for i, fp in enumerate(tqdm(self.filepaths, desc="Preload images")):
            images.append(
                np.stack(
                    [self.read_image(img_type[i]) for img_type in self.image_type_paths],
                    0,
                )
            )
        
        return images

    def get_image_paths(self):
        idxs = list(range(len(self.image_type_paths[0])))
        imgs = []

        for img_type in self.image_type_paths:
            types = []
            for idx in idxs:
                types.append(img_type[idx])

            imgs.append(types)

        return (idxs, *imgs)

    def get_image_path_from_indices(self, idxs, allow_oob=False):
        if isinstance(idxs, (list, np.ndarray)):
            imgs = []
            for img_type in self.image_type_paths:
                types = []

                for idx in idxs:
                    if idx >= len(img_type) and allow_oob:
                        # Allow for non-existing images.
                        types.append("")
                    else:
                        types.append(img_type[idx])
                imgs.append(types)

            return (idxs, *imgs)
        else:
            import tensorflow as tf

            if isinstance(idxs, tf.Tensor):
                types = []
                for img_type in self.image_type_paths:
                    if allow_oob:
                        max_idx = tf.reduce_max(idxs)
                        img_type = tf.concat(
                            [img_type, tf.repeat(tf.convert_to_tensor([""]), max_idx - img_type.shape[0], axis=0)],
                            axis=0
                        )
                    types.append(tf.gather_nd(img_type, tf.reshape(idxs, (-1, 1))))

                imgs = tf.stack(types, 0)

                type_list = tf.unstack(imgs, num=len(self.image_type_paths))
                return (idxs, *type_list)

    # def get_dataset_returns_for_indices(self, indices):
    #     return DatasetReturn(
    #         img_idx=indices,
    #         image_list=[
    #             [self.read_image(img_type[idx]) for idx in indices]
    #             for img_type in self.image_type_paths
    #         ],
    #     )

    def __getitem__(self, idx):
        
        return (
            idx,
            np.stack(
                [self.read_image(img_type[idx]) for img_type in self.image_type_paths],
                0,
            ),
        )
