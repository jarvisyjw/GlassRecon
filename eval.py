import argparse
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from alignment_functions import global_alignment, local_alignment

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def load_depth_from_file(file_path, depth_scale=1.0):
    """
    Load depth map from a file (npy, png, jpg) and apply scaling.
    """
    if file_path.endswith('.npy'):
        depth = np.load(file_path).astype(np.float32)
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        depth = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to load: {file_path}")
        depth = depth.astype(np.float32)
    else:
        raise ValueError(f"Unsupported format: {file_path}")
    depth = depth * depth_scale
    return depth

def find_file_by_basename(folder, basename, extensions):
    """
    Return the first file in 'folder' whose name starts with 'basename'
    and ends with one of the given extensions.
    """
    for ext in extensions:
        candidate = os.path.join(folder, basename + ext)
        if os.path.exists(candidate):
            return candidate
    return None

def compute_metrics(pred, gt, threshold=1.25):
    """
    Compute AbsRel and accuracy (δ < threshold).
    """
    pred = pred.squeeze()
    gt = gt.squeeze()

    mask = (gt > 1e-6) & (pred > 1e-6)
    mask = mask & ~np.isnan(gt) & ~np.isinf(gt)
    mask = mask & ~np.isnan(pred) & ~np.isinf(pred)

    if mask.sum() == 0:
        return np.nan, np.nan

    pred_masked = pred[mask]
    gt_masked = gt[mask]

    rel = np.abs(gt_masked - pred_masked) / gt_masked
    abs_rel = np.mean(rel)

    ratio = np.maximum(pred_masked / gt_masked, gt_masked / pred_masked)
    acc = (ratio < threshold).mean()

    return abs_rel, acc


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Evaluation')
    parser.add_argument('--image-folder', type=str, required=True,
                        help='Folder with RGB images (jpg/png) - used to get basenames')
    parser.add_argument('--pred-folder', type=str, required=True,
                        help='Folder with predicted depth maps')
    parser.add_argument('--sensor-depth-folder', type=str, required=True,
                        help='Folder with raw sensor depth (for alignment)')
    parser.add_argument('--gt-depth-folder', type=str, required=True,
                        help='Folder with ground truth depth (for evaluation)')
    parser.add_argument('--inverse-depth', action='store_true',
                        help='Prediction is inverse depth')
    parser.add_argument('--depth-scale', type=float, default=1.0,
                        help='Scale to convert raw values to meters')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Supported extensions (depth files can be npy, png, jpg/jpeg)
    depth_exts = ['.npy', '.png', '.jpg', '.jpeg']
    image_exts = ['.jpg', '.jpeg', '.png']

    # Get all image files and extract basenames
    image_names = []
    for f in os.listdir(args.image_folder):
        low = f.lower()
        for ext in image_exts:
            if low.endswith(ext):
                basename = f[:-len(ext)]
                image_names.append(basename)
                break

    if not image_names:
        raise RuntimeError(f"No images found in {args.image_folder}")

    results = []  # (name, global_absrel, global_acc, global_scale, global_shift,
                 #        ransac_absrel, ransac_acc, ransac_scale, ransac_shift)

    for basename in tqdm(image_names, desc="Evaluating"):
        # Locate corresponding files
        pred_path = find_file_by_basename(args.pred_folder, basename, depth_exts)
        sensor_path = find_file_by_basename(args.sensor_depth_folder, basename, depth_exts)
        gt_path = find_file_by_basename(args.gt_depth_folder, basename, depth_exts)

        if not (pred_path and sensor_path and gt_path):
            print(f"Warning: {basename} missing files, skipping")
            continue

        try:
            # Load
            pred = load_depth_from_file(pred_path, 1.0)
            sensor = load_depth_from_file(sensor_path, args.depth_scale)
            gt_eval = load_depth_from_file(gt_path, args.depth_scale)

            # Resize pred to match sensor size
            if pred.shape != sensor.shape:
                pred = cv2.resize(pred, (sensor.shape[1], sensor.shape[0]), interpolation=cv2.INTER_CUBIC)

            # Prepare tensors
            if args.inverse_depth:
                target = 1.0 / torch.from_numpy(sensor)
            else:
                target = torch.from_numpy(sensor)
            pred_t = torch.from_numpy(pred)

            mask = torch.from_numpy((sensor > 0.01) & (sensor < 100))
            target[torch.isinf(target) | torch.isnan(target)] = 0
            pred_t[torch.isinf(pred_t) | torch.isnan(pred_t)] = 0
            mask &= (target > 0) & (pred_t > 0)

            # --- Global (least squares) ---
            scale, shift = global_alignment(pred_t[None], target[None], mask[None])
            scale, shift = scale.item(), shift.item()
            aligned_global = scale * pred_t + shift
            if args.inverse_depth:
                aligned_global = 1.0 / aligned_global
            aligned_global = aligned_global.numpy()
            aligned_global = np.clip(aligned_global, 0, 100)

            # --- Local (RANSAC) ---
            scale_r, shift_r = local_alignment(pred_t[None], target[None], mask[None])
            scale_r, shift_r = scale_r.item(), shift_r.item()
            aligned_ransac = scale_r * pred_t + shift_r
            if args.inverse_depth:
                aligned_ransac = 1.0 / aligned_ransac
            aligned_ransac = aligned_ransac.numpy()
            aligned_ransac = np.clip(aligned_ransac, 0, 100)

            # Resize aligned predictions to GT size if different
            if aligned_global.shape != gt_eval.shape:
                aligned_global = cv2.resize(aligned_global, (gt_eval.shape[1], gt_eval.shape[0]), interpolation=cv2.INTER_LINEAR)
                aligned_ransac = cv2.resize(aligned_ransac, (gt_eval.shape[1], gt_eval.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Metrics
            absrel_g, acc_g = compute_metrics(aligned_global, gt_eval)
            absrel_r, acc_r = compute_metrics(aligned_ransac, gt_eval)

            results.append((basename, absrel_g, acc_g, scale, shift, absrel_r, acc_r, scale_r, shift_r))

        except Exception as e:
            print(f"Error with {basename}: {e}")

    # Produce output file
    valid = [r for r in results if not (np.isnan(r[1]) or np.isnan(r[5]))]
    if not valid:
        print("No valid results")
        exit(0)

    mean_absrel_g = np.mean([r[1] for r in valid])
    mean_acc_g    = np.mean([r[2] for r in valid])
    mean_absrel_r = np.mean([r[5] for r in valid])
    mean_acc_r    = np.mean([r[6] for r in valid])

    out_file = os.path.join(args.outdir, "eval_results.txt")
    with open(out_file, 'w') as f:
        f.write("Overall Results:\n")
        f.write(f"Global (LS) : AbsRel = {mean_absrel_g:.6f}, Acc = {mean_acc_g:.6f}\n")
        f.write(f"Local (RANSAC): AbsRel = {mean_absrel_r:.6f}, Acc = {mean_acc_r:.6f}\n\n")
        f.write("Per-image results (filename, global_absrel, global_acc, global_scale, global_shift, ransac_absrel, ransac_acc, ransac_scale, ransac_shift)\n")
        for r in results:
            line = f"{r[0]}, {r[1]}, {r[2]}, {r[3]}, {r[4]}, {r[5]}, {r[6]}, {r[7]}, {r[8]}\n"
            f.write(line)

    print(f"Done. Results saved to {out_file}")
    print(f"Processed {len(valid)} / {len(results)} samples successfully.")