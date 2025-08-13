import numpy as np

import json
from trellis.utils.icp import compute_icp_metrics
import os
import open3d as o3d




def load_pred(object_pred) -> o3d.geometry.TriangleMesh:
    return o3d.io.read_triangle_mesh(object_pred)

def load_gt(object_gt, scale=1.0) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(object_gt)
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * scale)
    return mesh


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--object_gt", type=str, default="")
    parser.add_argument("--object_pred", type=str, default="")
    parser.add_argument("--seq_name", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")


    args = parser.parse_args()
    from easydict import EasyDict

    args = EasyDict(vars(args))
    return args

def main():
    args = parse_args()
    
    data_pred = load_pred(args.object_pred)
    data_gt = load_gt(args.object_gt, scale=0.1)

        
    
    seq_name = args.seq_name
    out_p = args.out_dir
    os.makedirs(out_p, exist_ok=True)


    # Initialize the metrics dictionaries
    metric_dict = {}
    # Evaluate each metric using the corresponding function

    best_cd, best_f5, best_f10 = compute_icp_metrics(
        data_gt, data_pred, num_iters=600, out_dir=out_p
    )
    # Dictionary to store mean values of metrics
    mean_metrics = {}
    metric_dict["cd_icp"] = best_cd
    metric_dict["f5_icp"] = best_f5 * 100.0
    metric_dict["f10_icp"] = best_f10 * 100.0
    # metric_dict["cd_icp_no_scale"] = best_cd_no_scale
    # metric_dict["f5_icp_no_scale"] = best_f5_no_scale * 100.0
    # metric_dict["f10_icp_no_scale"] = best_f10_no_scale * 100.0
    # metric_dict["scale"] = scale

    # Print out the mean of each metric and store the results
    for metric_name, values in metric_dict.items():
        mean_value = float(
            np.nanmean(values)
        )  # Convert mean value to native Python float
        mean_metrics[metric_name] = mean_value

    # sort by key
    mean_metrics = dict(sorted(mean_metrics.items(), key=lambda item: item[0]))

    for metric_name, mean_value in mean_metrics.items():
        print(f"{metric_name.upper()}: {mean_value:.2f}")

    # Define the file paths
    json_path = out_p + "/metric.json"
    npy_path = out_p + "/metric_all.npy"

    from datetime import datetime

    current_time = datetime.now()
    time_str = current_time.strftime("%m-%d %H:%M")
    mean_metrics["timestamp"] = time_str
    mean_metrics["seq_name"] = seq_name
    print("Units: CD (cm), F-score (percentage)")

    # Save the mean_metrics dictionary to a JSON file with indentation
    with open(json_path, "w") as f:
        json.dump(mean_metrics, f, indent=4)
        print(f"Saved mean metrics to {json_path}")

    # Save the metric_all numpy array
    np.save(npy_path, metric_dict)
    print(f"Saved metric_all numpy array to {npy_path}")


if __name__ == "__main__":
    main()
