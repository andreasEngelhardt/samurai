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


def add_args(parser):
    parser.add_argument(
        "--datadir",
        required=True,
        type=str,
        help="Path to dataset location.",
    )

    parser.add_argument(
        "--max_resolution_dimension",
        type=int,
        default=800,
        help="Scales a image so the maximum resolution is at most the specified value",
    )

    parser.add_argument(
        "--test_holdout", type=int, default=16, help="Test holdout stride"
    )

    parser.add_argument(
        "--sparsity", type=int, default=0,
        choices=[0, 4, 6, 8, 16, 24],
        help=("Number of views to use from the sparse datasets. 0 disables sparsity. "
              "The data variants need to be precomputed at this point.")
    )

    parser.add_argument(
        "--noise_on_gt_poses",
        type=float,
        default=0.0,
        help="Add random noise to pose initialization with gt data.."
    )

    parser.add_argument(
        "--use_test_img_id_file",
        action="store_true",
        help="Read a sidecar file named test_img_id.txt to determine image ids for testing. This is used for neroic data, for example."
    )
    parser.add_argument(
        "--noise_keep_lookat",
        action="store_true",
        help="Keep the lookat directions when applying noise to gt."
    )
    parser.add_argument(
        "navi_version",
        type=str,
        default="v0.3",
        help="Version number of navi dataset."
    )
    parser.add_argument(
        "--coordinate_scale",
        type=float,
        default=1.0,
        help="Scaling factor on camera poses coordinates."
    )

    parser.add_argument("--dataset", choices=["samurai", "nerd", "navi", "orb"], default="samurai")
    parser.add_argument("--load_gt_poses", action="store_true")
    parser.add_argument("--load_gt_focals", action="store_true")
    parser.add_argument("--canonical_pose", type=int, default=0)
    parser.add_argument(
        "--gt_poses_format",
        choices=["blender", "colmap", "colmap_neroic", "colmap_orb"],
        default="blender"
    )

    return parser
