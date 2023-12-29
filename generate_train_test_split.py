# Generate a test_img_id.txt file for a fixed train_test split.
import argparse
import glob
import os
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="Percentage of training data in range 0-1."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        help="Path to datasety."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="navi",
        help="Dataset type. Default to 'navi'."
    )

    return parser


def save_to_file(fp, data):
    data_str = " ".join(str(x) for x in data)
    with open(fp, "w") as fi:
        fi.write(data_str)
    print(f"Saved to {fp}")


def main(arg_parser):
    args = arg_parser.parse_args()

    if args.dataset == "navi":
        image_subdir = "images"
    else:
        image_subdir = "image"

    for root, dirs, files in os.walk(args.root_dir):
        path = root.split(os.sep)
        if args.dataset == "navi":
            cond = "wild_set" in path
        else:
            cond = True
        if cond:
            for d in dirs:
                if d == image_subdir:
                    img_list = glob.glob(
                        os.path.join(root, d, "*.jpg")
                    )
                    img_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in img_list]
                    print(f"Found {len(img_ids)} images at {os.path.join(root, d)}: {img_ids}")
                    # Lets split.
                    num_train = round(len(img_ids) * args.train)
                    test_interval = int(round(len(img_ids) / (len(img_ids)-num_train)))
                    img_ids = np.sort(img_ids)
                    test_idx = img_ids[::test_interval]
                    if args.dataset == "navi":
                        save_path = os.path.join(
                            os.path.dirname(root.rstrip("/")),
                            "test_img_id.txt"    
                        )
                    else:
                        save_path = os.path.join(
                            root,
                            "test_img_id.txt"    
                        )
                    save_to_file(save_path, test_idx)
                    print(f"Selected {test_idx} for test set.")
        

if __name__ == "__main__":
    parser = get_args()

    main(parser)
