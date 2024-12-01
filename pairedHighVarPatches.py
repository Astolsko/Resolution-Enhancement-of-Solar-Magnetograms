import cv2
import numpy as np

# Load LR and HR images
soho_img_1k = r'renamedsoho\20100501_12000_M_512.jpg'
sdo_img_4k = r'renamedsdo\20100501_12000_M_512.jpg'

# Use grayscale for variance calculation
lr_image = cv2.imread(soho_img_1k, cv2.IMREAD_GRAYSCALE)
hr_image = cv2.imread(sdo_img_4k, cv2.IMREAD_GRAYSCALE)

# Check if images are loaded
if lr_image is None or hr_image is None:
    raise ValueError("One or both images could not be loaded. Check file paths.")

patch_size_lr = 128  # Low-resolution patch size
scaling_factor = 4  # Scaling factor (4096 / 1024)
patch_size_hr = patch_size_lr * scaling_factor  # High-resolution patch size

patch_variances_lr = []
patch_data_lr = {}  # To store LR patches by index
patch_data_hr = {}  # To store HR patches by index

i = 1  # Patch counter

# Loop through all patches, but skip the top row, bottom row, and side patches
for y_lr in range(128, lr_image.shape[0] - patch_size_lr, patch_size_lr):  # Adjust range
    for x_lr in range(128, lr_image.shape[1] - patch_size_lr, patch_size_lr):  # Adjust range

        # Skip the 4 specific corner patches
        if not (((x_lr == 128 and y_lr in [128, 768]) or (x_lr == 768 and y_lr in [128, 768]))):
            # Extract the LR patch
            lr_patch = lr_image[y_lr:y_lr + patch_size_lr, x_lr:x_lr + patch_size_lr]

            # Calculate HR patch coordinates and extract it
            x_hr = x_lr * scaling_factor
            y_hr = y_lr * scaling_factor

            if y_hr + patch_size_hr <= hr_image.shape[0] and x_hr + patch_size_hr <= hr_image.shape[1]:
                hr_patch = hr_image[y_hr:y_hr + patch_size_hr, x_hr:x_hr + patch_size_hr]

                # Calculate the variance of the LR patch
                patch_variance_lr = np.var(lr_patch)

                # Store the variance along with the patch index
                patch_variances_lr.append((patch_variance_lr, i))

                # Save the patches in a dictionary
                patch_data_lr[i] = lr_patch
                patch_data_hr[i] = hr_patch

                i += 1

# Sort the patches by variance in descending order
patch_variances_lr.sort(reverse=True, key=lambda x: x[0])

# Extract the top 5 patches with the highest variance
top_5_patches_lr = patch_variances_lr[:5]

# Save the top 5 patches
for variance, patch_index in top_5_patches_lr:
    patch = patch_data_lr[patch_index]
    if patch is not None and patch.size > 0:  # Check if patch is valid
        cv2.imwrite(f'lr_patch_{patch_index}.jpg', patch)

for variance, patch_index in top_5_patches_lr:
    patch = patch_data_hr[patch_index]
    if patch is not None and patch.size > 0:  # Check if patch is valid
        cv2.imwrite(f'hr_patch_{patch_index}.jpg', patch)

print("Top 5 patches with highest variance in LR image:")
for variance, patch_index in top_5_patches_lr:
    print(f"Patch {patch_index} - Variance: {variance}")
