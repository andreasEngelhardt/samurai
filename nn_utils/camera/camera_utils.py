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


import tensorflow as tf
from typing import Tuple
import nn_utils.math_utils as math_utils
import numpy as np


def stable_invert_c2i(c2i):
    fx = c2i[..., 0, 0]
    fy = -c2i[..., 1, 1]  # fy is inverted

    px = -c2i[..., 0, -1]  # px also
    py = -c2i[..., 1, -1]  # same for py

    o = tf.zeros_like(fx)
    l = tf.ones_like(fx)

    fx_recp = tf.math.reciprocal_no_nan(fx)
    fy_recp = tf.math.reciprocal_no_nan(fy)

    px_fx = tf.math.divide_no_nan(px, fx)
    py_fy = tf.math.divide_no_nan(py, fy)

    i2c = tf.stack(
        [
            tf.stack([fx_recp, o, -px_fx], -1),
            tf.stack([o, -fy_recp, py_fy], -1),
            tf.stack([o, o, -l], -1),
        ],
        -2,
    )

    return i2c


def stable_invert_c2w(c2w):
    r = c2w[..., :3, :3]
    t = c2w[..., :3, 3]

    idxs = list(range(len(r.shape)))
    transpose_idxs = [*idxs[:-2], idxs[-1], idxs[-2]]

    r_inv = tf.transpose(r, transpose_idxs)
    t_inv = math_utils.matrix_vector(r_inv, t)

    w2c = tf.concat([r_inv, -t_inv[..., None]], -1)
    w2c = math_utils.convert3x4_4x4(w2c)

    return w2c


def build_c2i(focal, principal):
    fx = focal[..., 0]
    fy = focal[..., 1]  # Shape is just batched

    px = principal[..., 0]
    py = principal[..., 1]

    o = tf.zeros_like(fx)
    l = tf.ones_like(fx)

    c2i = tf.stack(
        [
            tf.stack([fx, o, -px], 1),
            tf.stack([o, -fy, -py], 1),
            tf.stack([o, o, -l], 1),
        ],
        1,
    )

    return c2i


def stack_direction_vectors_to_matrix(x, y, z):
    tf.debugging.assert_shapes(
        [
            (x, (..., 3)),
            (y, (..., 3)),
            (z, (..., 3)),
        ]
    )
    return tf.stack([x, y, z], -2)


def build_look_at_matrix(eye_pos, center, up_rotation=0):
    w2c_direction = math_utils.normalize(eye_pos - center)

    value = tf.ones_like(eye_pos[..., :1]) * up_rotation

    # The regular up vector. Here, rotation 0 means default up -> Rotate around z axis
    up_regular = tf.concat(
        (
            tf.sin(value / 2),
            tf.cos(value / 2),
            tf.zeros_like(value),
        ),
        -1,
    )
    # The camera is at the north pole of the coordinate system. We now rotate around y axis
    up_north = tf.concat((tf.zeros_like(value), tf.sin(value), -tf.cos(value)), -1)
    # The camera is at the north pole of the coordinate system.
    up_south = tf.concat((tf.zeros_like(value), tf.sin(value), tf.cos(value)), -1)

    # Calculate which pole up location we should use
    reg_w2c_dot = math_utils.dot(w2c_direction, up_regular)
    up = tf.where(
        reg_w2c_dot >= 0.95,
        up_north,
        tf.where(reg_w2c_dot <= -0.95, up_south, up_regular),
    )

    assert up.shape == up_regular.shape

    camera_right = math_utils.normalize(math_utils.cross(up, w2c_direction))
    camera_up = math_utils.normalize(math_utils.cross(w2c_direction, camera_right))

    R = stack_direction_vectors_to_matrix(camera_right, camera_up, w2c_direction)

    return tf.linalg.inv(R)


def create_random_camera_permutations(
    t, scatter_div=5, distance_div=10, center_var=0.05, variants=400, random_up=True
):
    # Add permutation dim
    t = math_utils.repeat(tf.expand_dims(t, 1), variants, 1)
    distance = math_utils.magnitude(t)

    t_noise = tf.random.normal((t.shape[0], variants, 3), stddev=distance / scatter_div)
    distance_var = tf.random.normal(
        (t.shape[0], variants, 1), mean=distance, stddev=distance / distance_div
    )
    t = math_utils.normalize(t + t_noise) * distance_var

    random_center = tf.random.uniform(t.shape, minval=-center_var, maxval=center_var)
    r = build_look_at_matrix(t, random_center, random_up=random_up)

    return t, r


def c2w_to_lookat(c2w):
    tf.debugging.assert_shapes(
        [
            (c2w, ("B", "4", "4")),
        ]
    )
    batch_shape = tf.shape(c2w)[0]
    dir_shape = tf.zeros((batch_shape, 1), dtype=tf.float32)
    dirs = tf.concat(
        [
            tf.zeros_like(dir_shape),
            tf.zeros_like(dir_shape),
            -tf.ones_like(dir_shape),
        ],
        -1,
    )  # B, 3
    rays_d = tf.reduce_sum(dirs[..., None, :] * c2w[..., :3, :3], -1)  # B, 3
    rays_o = tf.broadcast_to(c2w[..., :3, -1], tf.shape(rays_d))  # B, 3

    ray_to_center_magn = math_utils.magnitude(rays_o)  # B, 1
    center = rays_o + math_utils.normalize(rays_d) * ray_to_center_magn

    return rays_o, center


def c2w_to_lookat_up(c2w: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Convert c2w matrix to eye, center and up parameters.

    Args:
        c2w (tf.Tensor): _description_

    Returns:
        Tuple: eye, center and up parameters matching input c2w as close as possible.
    """
    # This is a little hacky but so far the most reliable way.
    eye, center = c2w_to_lookat(c2w)

    r_flat = build_look_at_matrix(eye, center, up_rotation=0)
    c2w_flat = r_t_to_c2w(r_flat, eye)

    angles_target = euler_from_rotation_matrix(c2w[..., :3, :3])
    angles_default_up = euler_from_rotation_matrix(c2w_flat[..., :3, :3])
    # Get difference of angles around z axis.
    up = (angles_default_up - angles_target)[..., 2] * 2
    
    return eye, center, up


def r_t_to_c2w(r, t):
    tf.debugging.assert_shapes(
        [
            (r, (..., 3, 3)),
            (t, (..., 3)),
        ]
    )

    c2w = tf.concat([r, t[..., None]], -1)
    c2w = math_utils.convert3x4_4x4(c2w)

    return c2w


def euler_from_rotation_matrix(rotation_matrix: tf.Tensor,
                         name: str = "euler_from_rotation_matrix") -> tf.Tensor:
    """Converts rotation matrices to Euler angles.

    Adapted from Tensorflow Graphics: https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/euler.py#L140-L213

    The rotation matrices are assumed to have been constructed by rotation around
    the $$x$$, then $$y$$, and finally the $$z$$ axis.

    Note:
    There is an infinite number of solutions to this problem. There are
    Gimbal locks when abs(rotation_matrix(2,0)) == 1, which are not handled.

    Note:
    In the following, A1 to An are optional batch dimensions.

    Args:
        rotation_matrix: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
        dimensions represent a rotation matrix.
        name: A name for this op that defaults to "euler_from_rotation_matrix".

    Returns:
        A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
        the three Euler angles.

    """

    def general_case(rotation_matrix, r20, eps_addition):
        """Handles the general case."""
        theta_y = -tf.asin(r20)
        sign_cos_theta_y = math_utils.nonzero_sign(tf.cos(theta_y))
        r00 = rotation_matrix[..., 0, 0]
        r10 = rotation_matrix[..., 1, 0]
        r21 = rotation_matrix[..., 2, 1]
        r22 = rotation_matrix[..., 2, 2]
        r00 = math_utils.nonzero_sign(r00) * eps_addition + r00
        r22 = math_utils.nonzero_sign(r22) * eps_addition + r22
        # cos_theta_y evaluates to 0 on Gimbal locks, in which case the output of
        # this function will not be used.
        theta_z = tf.atan2(r10 * sign_cos_theta_y, r00 * sign_cos_theta_y)
        theta_x = tf.atan2(r21 * sign_cos_theta_y, r22 * sign_cos_theta_y)
        angles = tf.stack((theta_x, theta_y, theta_z), axis=-1)
        return angles

    def gimbal_lock(rotation_matrix, r20, eps_addition):
        """Handles Gimbal locks."""
        r01 = rotation_matrix[..., 0, 1]
        r02 = rotation_matrix[..., 0, 2]
        sign_r20 = math_utils.nonzero_sign(r20)
        r02 = math_utils.nonzero_sign(r02) * eps_addition + r02
        theta_x = tf.atan2(-sign_r20 * r01, -sign_r20 * r02)
        theta_y = -sign_r20 * tf.constant(np.pi / 2.0, dtype=r20.dtype)
        theta_z = tf.zeros_like(theta_x)
        angles = tf.stack((theta_x, theta_y, theta_z), axis=-1)
        return angles

    with tf.name_scope(name):
        rotation_matrix = tf.convert_to_tensor(value=rotation_matrix)

        tf.debugging.assert_shapes(
            [
                (rotation_matrix, (..., 3, 3)),
            ]
        )

        r20 = rotation_matrix[..., 2, 0]
        eps_addition = math_utils.select_eps_for_addition(rotation_matrix.dtype)
        general_solution = general_case(rotation_matrix, r20, eps_addition)
        gimbal_solution = gimbal_lock(rotation_matrix, r20, eps_addition)
        is_gimbal = tf.equal(tf.abs(r20), 1)
        gimbal_mask = tf.stack((is_gimbal, is_gimbal, is_gimbal), axis=-1)
        return tf.where(gimbal_mask, gimbal_solution, general_solution)

