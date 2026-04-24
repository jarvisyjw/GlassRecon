import torch
import torch.nn.functional as F

def global_alignment(prediction, target, mask):
    """_summary_

    Args:
        prediction (Tensor): disparity prediction, NxHxW
        target (Tensor): ground truth disparity, NxHxW
        mask (Tensor): invalid pixel mask, NxHxW

    Returns:
        x_0 (torch.float): scale factor
        x_1 (torch.float): shift factor
    """

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # print(torch.argwhere(torch.isnan(target)))
    # print(torch.argwhere(torch.isinf(target)))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # print(b_0)
    # print(det)
    # A needs to be a positive definite matrix.
    valid = det > 0 # mask out invalid values

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    print(f"scale: {x_0}", f"shift: {x_1}")

    return x_0, x_1


import torch

def local_alignment(
    predictions, targets, masks, 
    num_row=4, num_col=4, num_points_per_partition=10, num_sub_iterations=100):
    """Compute scale and shift using RANSAC for a batch of images with partition selection.

    Args:
        predictions (Tensor): disparity predictions, NxHxW.
        targets (Tensor): ground truth disparities, NxHxW.
        masks (Tensor): invalid pixel masks, NxHxW. 1 for valid pixels, 0 for invalid ones.
        num_row (int): number of rows for image partitioning (default: 4).
        num_col (int): number of columns for image partitioning (default: 4).
        num_points_per_partition (int): number of points sampled per partition for each RANSAC sub-iteration (default: 10).
        num_sub_iterations (int): number of sub-iterations to run per partition (default: 100).

    Returns:
        scales (Tensor): scale factors for each image in the batch, size: N.
        shifts (Tensor): shift factors for each image in the batch, size: N.
    """
    N, H, W = predictions.shape
    device = predictions.device
    
    # Initialize outputs
    scales = torch.ones(N, device=device)
    shifts = torch.zeros(N, device=device)
    
    # Precompute flat versions for efficient indexing
    predictions_flat = predictions.view(N, -1)
    targets_flat = targets.view(N, -1)
    masks_flat = masks.view(N, -1)
    
    # Precompute partition information
    row_step = H // num_row
    col_step = W // num_col
    
    for i in range(N):
        # Get valid indices for current image
        valid_mask_flat = masks_flat[i] > 0
        valid_indices_1d = torch.nonzero(valid_mask_flat, as_tuple=True)[0]
        
        if len(valid_indices_1d) < num_points_per_partition:
            print(f"Image {i}: Not enough valid points, using fallback")
            continue
        
        # Initialize best parameters
        best_total_error = float('inf')
        best_scale = 1.0
        best_shift = 0.0
        
        # Iterate over all partitions
        for r in range(num_row):
            for c in range(num_col):
                start_row, end_row = r * row_step, min((r + 1) * row_step, H)
                start_col, end_col = c * col_step, min((c + 1) * col_step, W)
                
                # Get valid indices in current partition
                partition_mask = torch.zeros(H, W, device=device, dtype=torch.bool)
                partition_mask[start_row:end_row, start_col:end_col] = True
                partition_valid_mask = partition_mask.view(-1) & valid_mask_flat
                partition_valid_indices = torch.nonzero(partition_valid_mask, as_tuple=True)[0]
                
                if len(partition_valid_indices) < num_points_per_partition:
                    continue
                
                # Best parameters for current partition
                best_partition_error = float('inf')
                best_partition_scale = 1.0
                best_partition_shift = 0.0
                
                # Multiple sub-iterations for current partition
                for _ in range(num_sub_iterations):
                    # Random sampling from partition
                    rand_indices = torch.randperm(len(partition_valid_indices), device=device)[:num_points_per_partition]
                    sampled_indices = partition_valid_indices[rand_indices]
                    
                    # Get sampled points
                    pred_samples = predictions_flat[i][sampled_indices]
                    target_samples = targets_flat[i][sampled_indices]
                    
                    # Solve linear system: target = scale * pred + shift
                    A = torch.stack([
                        pred_samples * pred_samples,
                        pred_samples,
                        torch.ones_like(pred_samples)
                    ], dim=1).sum(dim=0)
                    
                    b = torch.stack([
                        pred_samples * target_samples,
                        target_samples
                    ], dim=1).sum(dim=0)
                    
                    # Construct linear system
                    A_matrix = torch.tensor([
                        [A[0], A[1]],
                        [A[1], A[2]]
                    ], device=device)
                    
                    b_vector = torch.tensor([b[0], b[1]], device=device)
                    
                    # Check determinant
                    det = A_matrix[0, 0] * A_matrix[1, 1] - A_matrix[0, 1] * A_matrix[1, 0]
                    if torch.abs(det) < 1e-8:
                        continue
                    
                    # Solve for scale and shift
                    try:
                        solution = torch.linalg.solve(A_matrix, b_vector)
                        scale, shift = solution[0], solution[1]
                    except:
                        continue
                    
                    # Calculate errors for all valid points
                    predicted_values = scale * predictions_flat[i] + shift
                    errors = torch.abs(predicted_values - targets_flat[i])
                    errors[~valid_mask_flat] = 0  # Mask invalid points
                    
                    total_error = errors.sum()
                    
                    # Update best partition model
                    if total_error < best_partition_error:
                        best_partition_error = total_error
                        best_partition_scale = scale
                        best_partition_shift = shift
                
                # Update global best model
                if best_partition_error < best_total_error:
                    best_total_error = best_partition_error
                    best_scale = best_partition_scale
                    best_shift = best_partition_shift
        
        # If no valid partition found, use fallback: all valid points
        if best_total_error == float('inf'):
            print(f"Image {i}: No valid partition found, using fallback with all valid points")
            valid_pred = predictions_flat[i][valid_mask_flat]
            valid_target = targets_flat[i][valid_mask_flat]
            
            if len(valid_pred) > 0:
                A = torch.stack([
                    valid_pred * valid_pred,
                    valid_pred,
                    torch.ones_like(valid_pred)
                ], dim=1).sum(dim=0)
                
                b = torch.stack([
                    valid_pred * valid_target,
                    valid_target
                ], dim=1).sum(dim=0)
                
                A_matrix = torch.tensor([
                    [A[0], A[1]],
                    [A[1], A[2]]
                ], device=device)
                
                b_vector = torch.tensor([b[0], b[1]], device=device)
                
                try:
                    solution = torch.linalg.solve(A_matrix, b_vector)
                    best_scale, best_shift = solution[0], solution[1]
                except:
                    best_scale, best_shift = 1.0, 0.0
        
        # Store results
        scales[i] = best_scale
        shifts[i] = best_shift
        
        print(f"Image {i}: Scale = {scales[i].item()}, Shift = {shifts[i].item()}")
    
    return scales, shifts
