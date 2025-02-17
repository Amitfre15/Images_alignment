import os
import math
import re
import pandas as pd
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian


OPENSLIDE_PATH = r"C:\Program Files\openslide-bin-4.0.0.3-windows-x64\openslide-bin-4.0.0.3-windows-x64\bin"
# DESIRED_MPP = 0.5
DESIRED_MPP = 1
SLIDE_PATCH_SIZE = 256
SLIDE_MPP = 0.242797397769517
SLIDE_HEIGHT = 204614
SLIDE_WIDTH = 89484
HE_THUMB_HEIGHT = 10230
HE_THUMB_WIDTH = 4473
IHC_THUMB_HEIGHT = 5115
IHC_THUMB_WIDTH = 2237
IHC_SCALE_PATCH_SIZE = round(SLIDE_PATCH_SIZE * (IHC_THUMB_WIDTH / SLIDE_WIDTH) / 0.25)

if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def slide_to_thumb_coord(slide_coord, slide_dimensions, thumb_size):
    """
    Converts a coordinate from the slide to the corresponding coordinate in the thumbnail.

    Parameters:
    - slide: The OpenSlide object.
    - thumb: The thumbnail image (PIL Image or similar).
    - slide_coord: A tuple (x, y) representing the coordinate on the slide.

    Returns:
    - A tuple (x, y) representing the corresponding coordinate on the thumbnail.
    """
    # Get dimensions of the thumbnail
    thumb_width, thumb_height = thumb_size

    # Get dimensions of the slide
    slide_width, slide_height = slide_dimensions

    # Calculate scale factors
    scale_x = thumb_width / slide_width
    scale_y = thumb_height / slide_height

    # Convert slide coordinates to thumbnail coordinates
    thumb_x = int(slide_coord[0] * scale_x)
    thumb_y = int(slide_coord[1] * scale_y)

    return thumb_x, thumb_y


def thumb_to_slide_coord(thumb_coord, slide_dimensions, thumb_size):
    """
    Converts a coordinate from the thumbnail to the corresponding coordinate in the slide.

    Parameters:
    - thumb: The thumbnail image (PIL Image or similar).
    - slide: The OpenSlide object.
    - thumb_coord: A tuple (x, y) representing the coordinate on the thumb.

    Returns:
    - A tuple (x, y) representing the corresponding coordinate on the slide.
    """
    # Get dimensions of the thumbnail
    thumb_width, thumb_height = thumb_size

    # Get dimensions of the slide
    slide_width, slide_height = slide_dimensions

    # Calculate scale factors
    scale_x = slide_width / thumb_width
    scale_y = slide_height / thumb_height

    # Convert slide coordinates to thumbnail coordinates
    slide_x = int(thumb_coord[0] * scale_x)
    slide_y = int(thumb_coord[1] * scale_y)

    return slide_x, slide_y


def rotate_point(x, y, cx, cy, angle_rad):
    # Translate point to the origin (relative to the center)
    translated_x = x - cx
    translated_y = -(y - cy)

    # Apply the rotation matrix
    rotated_x = translated_x * math.cos(angle_rad) - translated_y * math.sin(angle_rad)
    rotated_y = translated_x * math.sin(angle_rad) + translated_y * math.cos(angle_rad)
    rotated_y = -rotated_y

    # Translate the point back
    # new_x = int(rotated_x + cx)
    # new_y = int(rotated_y + cy)

    # return new_x, new_y
    return rotated_x, rotated_y


def rotate_coordinates(corners, angle, image_center=None):
    """
    Rotates the coordinates of the four corners of an image by a given angle.

    Parameters:
    - x1, y1: Coordinates of the top-left corner.
    - x2, y2: Coordinates of the top-right corner.
    - x3, y3: Coordinates of the bottom-right corner.
    - x4, y4: Coordinates of the bottom-left corner.
    - angle: Rotation angle in degrees (counterclockwise).
    - image_center: (cx, cy) The center of the image for rotation. If None, it is calculated as the center of the given coordinates.

    Returns:
    - A list of tuples representing the new coordinates of the corners after rotation.
    """
    # Convert the angle to radians
    angle_rad = math.radians(angle)

    # Calculate the center of the image if not provided
    if image_center is None:
        cx = (corners[0] + corners[2]) / 2
        cy = (corners[1] + corners[5]) / 2
    else:
        cx, cy = image_center

    # Rotate all four corners
    new_x1, new_y1 = rotate_point(corners[0], corners[1], cx, cy, angle_rad)
    new_x2, new_y2 = rotate_point(corners[2], corners[3], cx, cy, angle_rad)
    new_x3, new_y3 = rotate_point(corners[4], corners[5], cx, cy, angle_rad)
    new_x4, new_y4 = rotate_point(corners[6], corners[7], cx, cy, angle_rad)

    return [(new_x1, new_y1), (new_x2, new_y2), (new_x3, new_y3), (new_x4, new_y4)]


def get_padded_slide_coords(slide_coords: list, slide_width: int, slide_height: int):
    # Calculate the original patch width and height
    patch_width = slide_coords[2] - slide_coords[0]
    patch_height = slide_coords[3] - slide_coords[1]

    # Calculate the padding using 2 scaling factor
    scaling_factor = 2
    padding_width = int((scaling_factor - 1) * patch_width / 2)
    padding_height = int((scaling_factor - 1) * patch_height / 2)

    # Add padding to the coordinates
    padded_slide_coords = (
        max(0, slide_coords[0] - padding_width),
        max(0, slide_coords[1] - padding_height),
        min(slide_width, slide_coords[2] + padding_width),
        min(slide_height, slide_coords[3] + padding_height)
    )

    return padded_slide_coords, patch_width, patch_height


def extract_padded_slide_patch(padded_slide_coords: list, slide, patch_width: int, patch_height: int,
                               rotation_angle: float):
    # Extract padded slide patch
    padded_slide_width = padded_slide_coords[2] - padded_slide_coords[0]
    padded_slide_height = padded_slide_coords[3] - padded_slide_coords[1]
    slide_patch = slide.read_region((padded_slide_coords[0], padded_slide_coords[1]), 0,
                                    (padded_slide_width, padded_slide_height))
    # slide_patch.show(title="Slide Patch")
    angle_rad = math.radians(rotation_angle)
    actual_pad_width = (slide_patch.width - patch_width) // 2
    actual_pad_height = (slide_patch.height - patch_height) // 2
    corners = [actual_pad_width, actual_pad_height,
               slide_patch.width - actual_pad_width, actual_pad_height,
               actual_pad_width, slide_patch.height - actual_pad_height,
               slide_patch.width - actual_pad_width, slide_patch.height - actual_pad_height]
    slide_patch = slide_patch.rotate(rotation_angle, expand=True)
    slide_patch.show(title="Slide Patch")

    return slide_patch, corners


def crop_padded_patch(corners: list, rotation_angle: float, slide_patch):
    # Crop the padded patch
    rotated_corners = rotate_coordinates(corners=corners, angle=rotation_angle)
    new_center = slide_patch.width // 2, slide_patch.height // 2
    xs = [rc[0] + new_center[0] for rc in rotated_corners]
    ys = [rc[1] + new_center[1] for rc in rotated_corners]
    small_x, big_x = min(xs), max(xs)
    small_y, big_y = min(ys), max(ys)
    crop_box_slide = (small_x, small_y, big_x, big_y)
    slide_patch = slide_patch.crop(crop_box_slide)

    return slide_patch


# Function to round mpp values to the nearest 1/(2n) or 1
def round_mpp(mpp_value):
    try:
        if abs(mpp_value - 1) < 0.1:
            return 1
        else:
            n = int(round(1 / (2 * mpp_value)))
            return 1 / (2 * n)
    except:
        return None


def get_mpp_origin(slide):
    x_origin = int(slide.properties.get('openslide.bounds-x'))
    y_origin = int(slide.properties.get('openslide.bounds-y'))
    mpp_x = float(slide.properties.get('openslide.mpp-x'))
    mpp_y = float(slide.properties.get('openslide.mpp-y'))

    return mpp_x, mpp_y, x_origin, y_origin


def extract_and_show_patch(slide, thumb, rotation_mat=None, rotation_angle=None, slide_coords=None, thumb_coords=None,
                           prefix=''):
    """
    Extracts patches from the slide and the thumbnail based on the given slide coordinates and shows them.

    Parameters:
    - slide: The OpenSlide object.
    - thumb: The thumbnail image (PIL Image or similar).
    - slide_coords: A tuple (x1, y1, x2, y2) representing the top-left and bottom-right coordinates on the slide.
    - rotation: A float representing the rotation angle.

    Returns:
    - None, but displays the extracted patches.
    """
    if slide_coords:
        # padded_slide_coords, patch_width, patch_height = get_padded_slide_coords(slide_coords=slide_coords,
        #                                                                          slide_width=slide.dimensions[0],
        #                                                                          slide_height=slide.dimensions[1])

        # Map the slide coordinates to thumbnail coordinates
        thumb_coord1 = slide_to_thumb_coord((slide_coords[0], slide_coords[1]),
                                            slide_dimensions=slide.dimensions, thumb_size=thumb.size)
        thumb_coord2 = slide_to_thumb_coord((slide_coords[2], slide_coords[3]),
                                            slide_dimensions=slide.dimensions, thumb_size=thumb.size)
        thumb_coords = list(thumb_coord1) + list(thumb_coord2)
        # the first slides were 90 degrees rotated
        # rotation_angle = (rotation_mat[thumb_coords[0], thumb_coords[1]][-1] / 100) - 90
        # rotation_angle = 0

        # slide_patch, corners = extract_padded_slide_patch(padded_slide_coords=padded_slide_coords, slide=slide,
        #                                                   patch_width=patch_width, patch_height=patch_height,
        #                                                   rotation_angle=rotation_angle)
        #
        # slide_patch = crop_padded_patch(corners=corners, rotation_angle=rotation_angle, slide_patch=slide_patch)
        slide_patch = slide.read_region((slide_coords[0], slide_coords[1]), 0,
                                        (slide_coords[2] - slide_coords[0], slide_coords[3] - slide_coords[1]))
    else:
        # Map the thumbnail coordinates to slide coordinates
        slide_coord1 = thumb_to_slide_coord(slide_dimensions=slide.dimensions, thumb_size=thumb.size,
                                            thumb_coord=(thumb_coords[0], thumb_coords[1]))
        slide_coord2 = thumb_to_slide_coord(slide_dimensions=slide.dimensions, thumb_size=thumb.size,
                                            thumb_coord=(thumb_coords[2], thumb_coords[3]))
        slide_coords = list(slide_coord1) + list(slide_coord2)
        # rotation_angle = rotation_mat[int(thumb_coords[0]), int(thumb_coords[1])][-1] / 100

        padded_slide_coords, patch_width, patch_height = get_padded_slide_coords(slide_coords=slide_coords,
                                                                                 slide_width=slide.dimensions[0],
                                                                                 slide_height=slide.dimensions[1])
        slide_patch, corners = extract_padded_slide_patch(padded_slide_coords=padded_slide_coords, slide=slide,
                                                          patch_width=patch_width, patch_height=patch_height,
                                                          rotation_angle=rotation_angle)
        slide_patch = crop_padded_patch(corners=corners, rotation_angle=rotation_angle, slide_patch=slide_patch)

        # slide_patch = slide.read_region((slide_coords[0], slide_coords[1]), 0,
        #                                 (slide_coords[2] - slide_coords[0], slide_coords[3] - slide_coords[1]))
        # slide_patch = slide_patch.rotate(rotation_angle, expand=True)

    # Extract thumb patches
    thumb_patch = thumb.crop((thumb_coords[0], thumb_coords[1], thumb_coords[2], thumb_coords[3]))

    # Convert slide patch to RGB (it might be RGBA, depending on the format)
    slide_patch_rgb = slide_patch.convert("RGB").resize((SLIDE_PATCH_SIZE, SLIDE_PATCH_SIZE))
    # slide_patch_rgb.show(title="Slide Patch")
    # thumb_patch = thumb_patch.rotate(rotation_angle, expand=True)

    # if prefix == 'ihc_':
    # Display both patches
    thumb_patch.show(title="Thumbnail Patch")
    slide_patch_rgb.show(title="Slide Patch")
    slide_patch_rgb.save(fp=os.path.join('slides_to_thumbs_output', f'{prefix}slide_patch.png'), format="PNG")
    thumb_patch.save(fp=os.path.join('slides_to_thumbs_output', f'{prefix}thumb_patch.png'), format="PNG")

    return thumb_coords


def extract_matching_slide_thumb_patches(he_slide_path: str, he_thumb_path, ihc_slide_path: str, ihc_thumb_path: str,
                                         rotation_mat_path: str, qupath_he_coords: tuple):
    h_e_slide = openslide.OpenSlide(he_slide_path)
    thumb = Image.open(he_thumb_path)

    mpp_x, mpp_y, x_origin, y_origin = get_mpp_origin(h_e_slide)

    # qupath_location = (16200, 42400)  # 21-3263 slides
    # openslide_location = (round(qupath_location[0] / mpp_x) + x_origin, round(qupath_location[1] / mpp_y) + y_origin)
    openslide_location = (round(qupath_he_coords[0] / mpp_x) + x_origin, round(qupath_he_coords[1] / mpp_y) + y_origin)
    mpp_scale_factor = DESIRED_MPP / mpp_x
    scaled_patch_size = int(SLIDE_PATCH_SIZE * mpp_scale_factor)

    slide_coords = tuple(list(openslide_location) + [cor + scaled_patch_size for cor in openslide_location])
    rotation_img = cv2.imread(rotation_mat_path, cv2.IMREAD_UNCHANGED)
    rotation_img = cv2.cvtColor(rotation_img, cv2.COLOR_BGR2RGB)
    rotation_mat = np.array(rotation_img).transpose(1, 0, 2)

    thumb_coords = extract_and_show_patch(h_e_slide, thumb, rotation_mat=rotation_mat, slide_coords=slide_coords)
    # minus 1 since the matrix holds rotation angles from HE to IHC, but we rotate IHC to HE
    rotation_angle = -1 * (rotation_mat[int(thumb_coords[0]), int(thumb_coords[1])][-1] / 100)
    # rotation_angle = 0
    thumb_center = [(thumb_coords[0] + thumb_coords[2]) // 2, (thumb_coords[1] + thumb_coords[3]) // 2]
    thumb_width, thumb_height = thumb_coords[2] - thumb_coords[0], thumb_coords[3] - thumb_coords[1]
    ihc_center = list(rotation_mat[thumb_center[0], thumb_center[1]][:2])

    # IHC
    ihc_slide = openslide.OpenSlide(ihc_slide_path)
    ihc_thumb = Image.open(ihc_thumb_path)
    x_ratio = ihc_thumb.width / thumb.width
    y_ratio = ihc_thumb.height / thumb.height
    corresp_ihc_thumb_coords = [max(0, ihc_center[0] - (thumb_width // 2) * x_ratio),
                                max(0, ihc_center[1] - (thumb_height // 2) * y_ratio),
                                min(ihc_thumb.width, ihc_center[0] + (thumb_width // 2) * x_ratio),
                                min(ihc_thumb.height,
                                    ihc_center[1] + (thumb_height // 2) * y_ratio)]

    extract_and_show_patch(ihc_slide, ihc_thumb, rotation_angle=rotation_angle, thumb_coords=corresp_ihc_thumb_coords,
                           prefix='ihc_')


# Extract slide names and coordinates from window_name
def parse_window_name(window_name):
    # Example format: "21-1518_1_5_d_32256_33024x_10240_12288y"
    match = re.match(r"(.*)_(\d+)_(\d+)x_(\d+)_(\d+)y", window_name)
    if match:
        slide_name = match.group(1)
        x_start, x_end, y_start, y_end = map(int, match.groups()[1:])
        x_end, y_end = x_end + SLIDE_PATCH_SIZE, y_end + SLIDE_PATCH_SIZE
        return slide_name, x_start, x_end, y_start, y_end
    else:
        raise ValueError(f"Invalid window_name format: {window_name}")


def find_matching_window_score(window_name: str, score_matrix_path: str, map_matrix_path: str):
    rounded_mpp = round_mpp(SLIDE_MPP)
    mpp_correction_factor = SLIDE_MPP / rounded_mpp

    slide_name, x_start, x_end, y_start, y_end = parse_window_name(window_name)
    x_start, x_end, y_start, y_end = tuple(map(lambda x: x * mpp_correction_factor / SLIDE_MPP, (x_start, x_end, y_start, y_end)))
    he_window_center = ((x_end + x_start) // 2, (y_end + y_start) // 2)
    he_thumb_window_center = slide_to_thumb_coord(slide_coord=he_window_center,
                                                  slide_dimensions=(SLIDE_WIDTH, SLIDE_HEIGHT),
                                                  thumb_size=(HE_THUMB_WIDTH, HE_THUMB_HEIGHT))

    score_mat = np.load(score_matrix_path)['arr_0']  # saved as .npz file
    map_img = cv2.imread(map_matrix_path, cv2.IMREAD_UNCHANGED)
    map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
    map_mat = np.array(map_img).transpose(1, 0, 2)

    ihc_thumb_window_center = map_mat[he_thumb_window_center[0], he_thumb_window_center[1]][:2]
    score = score_mat[ihc_thumb_window_center[0], ihc_thumb_window_center[1]]

    return score


def show_score_matrix(score_matrix, slide_name):
    plt.figure(figsize=(10, 8))
    plt.title(f"Score Matrix for Slide: {slide_name}")
    # score_matrix[1000:2000, 1000:2000] = 255
    plt.imshow(score_matrix, cmap='gray', interpolation='none', vmin=-1, vmax=4)
    png_path = os.path.join('slides_and_thumbs', f'{slide_name}_score_mat.png')
    if not os.path.exists(png_path):
        plt.imsave(png_path, arr=score_matrix, cmap='gray', vmin=-1, vmax=4)
    plt.colorbar(label='Score')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.tight_layout()
    plt.show()


def gaussian_2d_kernel(shape, sigma=1, window_amp: bool = False, patch_num: int = None) -> torch.tensor:
    """
    Generate a 2D Gaussian kernel for a non-square shape.

    Parameters:
        shape (tuple): The shape of the kernel (height, width).
        sigma (float): Standard deviation of the Gaussian along the x-axis.

    Returns:
        torch.tensor: A 2D Gaussian kernel.
    """
    height, width = shape
    gaussian_x = gaussian(width, std=sigma)  # 1D Gaussian along the x-axis
    gaussian_y = gaussian(height, std=sigma)  # 1D Gaussian along the y-axis

    kernel = np.outer(gaussian_y, gaussian_x)  # Create 2D Gaussian from 1D arrays
    kernel /= kernel.sum()  # Normalize to ensure the kernel sums to 1

    if window_amp and patch_num is not None:
        ker_center = kernel[height // 2, width // 2]
        desired_center_val = 1 / patch_num
        center_factor = desired_center_val / ker_center
        kernel *= center_factor
    return torch.tensor(kernel)


def create_score_matrices(slide_scores_csv: str):
    rounded_mpp = round_mpp(SLIDE_MPP)
    mpp_correction_factor = SLIDE_MPP / rounded_mpp

    scores_df = pd.read_csv(slide_scores_csv)

    # Create a directory to save score matrices
    output_dir = "score_matrices"
    os.makedirs(output_dir, exist_ok=True)

    # Parse window_name and create columns for easy processing
    scores_df[['slide_name', 'x_start', 'x_end', 'y_start', 'y_end']] = scores_df['window_name'].apply(
        lambda x: pd.Series(parse_window_name(x))
    )

    # Group by slide
    for slide_name, group in scores_df.groupby('slide_name'):
        # Initialize the score matrix with NaN
        score_matrix = np.full((IHC_THUMB_HEIGHT, IHC_THUMB_WIDTH), np.nan, dtype=np.float16)

        # Fill the score matrix with scores from each window
        for _, row in group.iterrows():
            score = row['score']
            x_start, x_end, y_start, y_end = row['x_start'], row['x_end'], row['y_start'], row['y_end']
            x_start, x_end, y_start, y_end = tuple(map(lambda x: x * mpp_correction_factor / SLIDE_MPP, (x_start, x_end, y_start, y_end)))
            x_start, y_start = slide_to_thumb_coord(slide_coord=(x_start, y_start),
                                                    slide_dimensions=(SLIDE_WIDTH, SLIDE_HEIGHT),
                                                    thumb_size=(IHC_THUMB_WIDTH, IHC_THUMB_HEIGHT))
            x_end, y_end = slide_to_thumb_coord(slide_coord=(x_end, y_end),
                                                slide_dimensions=(SLIDE_WIDTH, SLIDE_HEIGHT),
                                                thumb_size=(IHC_THUMB_WIDTH, IHC_THUMB_HEIGHT))
            score_matrix[x_start:x_end, y_start:y_end] = score

        show_score_matrix(score_matrix=score_matrix, slide_name=slide_name)

        # Save the score matrix as a .npz file (NumPy compressed format)
        npz_output_path = os.path.join(output_dir, f"{slide_name}_score_matrix.npz")
        np.savez_compressed(npz_output_path, score_matrix)
        # loaded_matrix = np.load(npz_output_path)['arr_0']

        print(f"Saved score matrix for slide: {slide_name} at {npz_output_path}")


def create_weighted_score_matrices(slide_scores_csv: str):
    rounded_mpp = round_mpp(SLIDE_MPP)
    mpp_correction_factor = SLIDE_MPP / rounded_mpp

    scores_df = pd.read_csv(slide_scores_csv)

    # Parse window_name and create columns for easy processing
    scores_df[['slide_name', 'x_start', 'x_end', 'y_start', 'y_end']] = scores_df['window_name'].apply(
        lambda x: pd.Series(parse_window_name(x))
    )

    # Group by slide
    for slide_name, group in scores_df.groupby('slide_name'):
        # Initialize score and weight matrices
        weighted_score_matrix = np.zeros((IHC_THUMB_HEIGHT, IHC_THUMB_WIDTH), dtype=np.float64)
        weight_matrix = np.zeros((IHC_THUMB_HEIGHT, IHC_THUMB_WIDTH), dtype=np.float64)

        # Fill the score matrix with scores from each window
        for _, row in group.iterrows():
            score = row['score']
            x_start, x_end, y_start, y_end = row['x_start'], row['x_end'], row['y_start'], row['y_end']
            x_start, x_end, y_start, y_end = tuple(map(lambda x: x * mpp_correction_factor / SLIDE_MPP, (x_start, x_end, y_start, y_end)))
            x_start, y_start = slide_to_thumb_coord(slide_coord=(x_start, y_start),
                                                    slide_dimensions=(SLIDE_WIDTH, SLIDE_HEIGHT),
                                                    thumb_size=(IHC_THUMB_WIDTH, IHC_THUMB_HEIGHT))
            x_end, y_end = slide_to_thumb_coord(slide_coord=(x_end, y_end),
                                                slide_dimensions=(SLIDE_WIDTH, SLIDE_HEIGHT),
                                                thumb_size=(IHC_THUMB_WIDTH, IHC_THUMB_HEIGHT))

            # Create a Gaussian window for the current region
            window_size = (x_end - x_start, y_end - y_start)
            gaussian_window = gaussian_2d_kernel(shape=window_size)

            # Apply the Gaussian window to the score
            weighted_region = gaussian_window * score

            try:
                # Add to the cumulative matrices
                weighted_score_matrix[x_start:x_end, y_start:y_end] += weighted_region
                weight_matrix[x_start:x_end, y_start:y_end] += gaussian_window
            except BaseException as e:
                print(e)

        # Normalize the weighted score matrix by the weight matrix
        final_score_matrix = np.divide(
            weighted_score_matrix,
            weight_matrix,
            out=np.zeros_like(weighted_score_matrix),
            where=weight_matrix > 0  # Avoid division by zero
        )

        show_score_matrix(score_matrix=final_score_matrix, slide_name=slide_name)

        # Save the score matrix as a .npz file (NumPy compressed format)
        compressed_score_matrix = final_score_matrix.astype(np.float16)
        npz_output_path = os.path.join(output_dir, f"{slide_name}_score_matrix.npz")
        np.savez_compressed(npz_output_path, compressed_score_matrix)
        # loaded_matrix = np.load(npz_output_path)['arr_0']
        # show_score_matrix(score_matrix=loaded_matrix, slide_name=slide_name)

        print(f"Saved score matrix for slide: {slide_name} at {npz_output_path}")


def gaussian_patches_matrix(weight_matrix: torch.tensor, gaus_window: torch.tensor, img_coords: torch.tensor):
    # Get the dimensions of the weight matrix
    w_height, w_width = weight_matrix.size()

    # Get the dimensions of the gaus window
    k_height, k_width = gaus_window.size()

    # Flatten img_coords for easier handling
    start_x = img_coords[:, :, 0].long().flatten()  # x_start
    start_y = img_coords[:, :, 1].long().flatten()  # y_start

    pad_pixels = int(IHC_SCALE_PATCH_SIZE * 0.25)
    # Create a grid for the small_matrix offsets
    x_range = torch.arange(-pad_pixels, k_height - pad_pixels).repeat(k_height).view(k_height,
                                                           k_height).t().flatten().to(device=start_x.device)  # Row indices for the small matrix
    y_range = torch.arange(-pad_pixels, k_height - pad_pixels).repeat(k_width).view(-1, 1).to(device=start_y.device)  # Column indices for the small matrix

    # Compute all region coordinates by adding offsets
    x_offsets = start_x.view(-1, 1, 1) + x_range  # Broadcast offsets for x
    y_offsets = start_y.view(-1, 1, 1) + y_range  # Broadcast offsets for y

    # Flatten offsets for direct indexing
    x_offsets = x_offsets.flatten()  # Shape: (n_regions * k_height * k_width)
    y_offsets = y_offsets.flatten()  # Shape: (n_regions * k_height * k_width)

    # Tile the small_matrix to match all regions
    gaus_window_tiled = gaus_window.repeat(len(start_x), 1,
                                             1).flatten().to(device=start_x.device)  # Shape: (n_regions * k_height * k_width)

    # Filter out-of-bound indices
    valid_mask = (x_offsets >= 0) & (x_offsets < w_height) & (y_offsets >= 0) & (y_offsets < w_width)

    # Apply the mask
    x_offsets = x_offsets[valid_mask].to(device=start_x.device)
    y_offsets = y_offsets[valid_mask].to(device=start_x.device)
    gaus_window_tiled = gaus_window_tiled[valid_mask].to(device=start_x.device)

    # Add the small_matrix values to the weight_matrix
    weight_matrix.index_put_((x_offsets, y_offsets), gaus_window_tiled, accumulate=True)
    return weight_matrix


def patch_weighted_score_matrices(weighted_score_matrix: torch.tensor, weight_matrix: torch.tensor,
                                  img_coords: torch.tensor, window_score: float):
    rounded_mpp = round_mpp(SLIDE_MPP)
    mpp_correction_factor = SLIDE_MPP / rounded_mpp

    img_coords = img_coords * mpp_correction_factor / SLIDE_MPP

    # Calculate scale factors
    scale = IHC_THUMB_WIDTH / SLIDE_WIDTH

    # Convert slide coordinates to thumbnail coordinates
    img_thumb_coords = (img_coords * scale).to(torch.int32)

    pad_pixels = int(IHC_SCALE_PATCH_SIZE * 0.25)
    gaus_win_side = IHC_SCALE_PATCH_SIZE + 2 * pad_pixels
    sq_gaus_window_size = (gaus_win_side, gaus_win_side)
    gaussian_window = gaussian_2d_kernel(shape=sq_gaus_window_size, window_amp=True, patch_num=img_coords.size(1))

    curr_weight_matrix = torch.zeros_like(weight_matrix).to(device=weight_matrix.device)
    curr_weight_matrix = gaussian_patches_matrix(weight_matrix=curr_weight_matrix, gaus_window=gaussian_window, img_coords=img_thumb_coords)
    weighted_score_matrix += curr_weight_matrix * window_score
    weight_matrix += curr_weight_matrix


def main():
    # find_matching_window_score(window_name=, score_matrix_dir=)
    # slide_scores_csv_path = os.path.join('excel_files', 'sw_slide_scores.csv')

    # sigma = 'sigma1'
    # weighted_score_mat_dir = os.path.join('weighted_score_matrices', sigma)

    slides_and_thumbs_path = os.path.join('slides_and_thumbs')
    npz_files = list(filter(lambda x: x.endswith('.npz'), os.listdir('slides_and_thumbs')))
    # for w_path in os.listdir(weighted_score_mat_dir):
    for w_path in npz_files:
        if w_path == "21-2989_1_1_m_score_matrix.npz":
            # full_mat_path = os.path.join(weighted_score_mat_dir, w_path)
            full_mat_path = os.path.join(slides_and_thumbs_path, w_path)
            loaded_matrix = np.load(full_mat_path)['arr_0']
            # show_score_matrix(loaded_matrix[3990: 4100, 480:590], slide_name='')
            show_score_matrix(loaded_matrix, slide_name=w_path.split('\\')[-1].split('_score')[0])

    # slide_scores_csv_path = os.path.join('excel_files', 'sw_stride1_slide_scores.csv')
    # create_weighted_score_matrices(slide_scores_csv=slide_scores_csv_path)

    # img_coords = torch.tensor([[[10240., 0.],
    #                       [10496., 0.],
    #                       [10496., 256.],
    #                       [10496., 1280.],
    #                       [10752., 0.],
    #                       [10752., 256.],
    #                       [11008., 0.],
    #                       [11008., 256.]]])
    # img_coords = torch.tensor([[[40192., 4864.],
    #                       [40192., 5120.],
    #                       [40192., 6656.],
    #                       [40192., 6912.],
    #                       [40448., 6912.],
    #                       [40704., 6912.],
    #                       [40960., 6912.],
    #                       [41216., 4608.],
    #                       [41216., 4864.],
    #                       [41216., 6912.],
    #                       [41472., 4864.],
    #                       [41472., 5120.],
    #                       [41472., 5376.],
    #                       [41472., 5888.],
    #                       [41472., 6144.],
    #                       [41472., 6400.],
    #                       [41472., 6656.],
    #                       [41472., 6912.],
    #                       [41728., 4608.],
    #                       [41728., 4864.],
    #                       [41728., 5120.],
    #                       [41728., 5376.],
    #                       [41728., 5632.],
    #                       [41728., 6144.],
    #                       [41728., 6400.],
    #                       [41728., 6656.],
    #                       [41728., 6912.],
    #                       [41984., 4608.],
    #                       [41984., 4864.],
    #                       [41984., 5120.],
    #                       [41984., 5376.],
    #                       [41984., 5632.],
    #                       [41984., 5888.],
    #                       [41984., 6144.],
    #                       [41984., 6400.],
    #                       [41984., 6656.],
    #                       [41984., 6912.],
    #                       [42240., 4608.],
    #                       [42240., 4864.],
    #                       [42240., 5120.],
    #                       [42240., 5376.],
    #                       [42240., 5632.],
    #                       [42240., 5888.],
    #                       [42240., 6144.],
    #                       [42240., 6400.],
    #                       [42240., 6656.],
    #                       [42240., 6912.],
    #                       [42496., 4608.],
    #                       [42496., 4864.],
    #                       [42496., 5120.],
    #                       [42496., 5376.],
    #                       [42496., 5632.],
    #                       [42496., 5888.],
    #                       [42496., 6144.],
    #                       [42496., 6400.],
    #                       [42496., 6656.],
    #                       [42496., 6912.]]])
    # weighted_score_matrix = torch.zeros((IHC_THUMB_HEIGHT, IHC_THUMB_WIDTH), dtype=torch.float64)
    # weight_matrix = torch.zeros((IHC_THUMB_HEIGHT, IHC_THUMB_WIDTH), dtype=torch.float64)
    # patch_weighted_score_matrices(weighted_score_matrix=weighted_score_matrix, weight_matrix=weight_matrix,
    #                               img_coords=img_coords, window_score=3)
    # patch_weighted_score_matrices(weighted_score_matrix=weighted_score_matrix, weight_matrix=weight_matrix,
    #                               img_coords=img_coords, window_score=2)
    #
    # final_score_matrix = np.divide(
    #     weighted_score_matrix.cpu().numpy(),
    #     weight_matrix.cpu().numpy(),
    #     out=np.zeros_like(weighted_score_matrix.cpu().numpy()),
    #     where=weight_matrix.cpu().numpy() > 0  # Avoid division by zero
    # )
    # pass

    # **** slide patches extraction ****
    # he_slide_path = os.path.join('slides_and_thumbs', '21-3263_1_1_e.mrxs')
    # he_thumb_path = os.path.join('slides_and_thumbs', '0235_0_thumb_21-3263_1_1_e.png')
    # ihc_slide_path = os.path.join('slides_and_thumbs', '21-3263_1_1_m.mrxs')
    # ihc_thumb_path = os.path.join('slides_and_thumbs', '0131_0_thumb_21-3263_1_1_m.png')
    # rotation_mat_path = os.path.join('slides_and_thumbs', 'map_HE_21-3263_1_1_e_to_Her2_21-3263_1_1_m.png')
    # qupath_he_coords = (16200, 42400)
    # extract_matching_slide_thumb_patches(he_slide_path=he_slide_path, he_thumb_path=he_thumb_path,
    #                                      ihc_slide_path=ihc_slide_path, ihc_thumb_path=ihc_thumb_path,
    #                                      rotation_mat_path=rotation_mat_path, qupath_he_coords=qupath_he_coords)


if __name__ == '__main__':
    main()


def indicative_patches():
    pass
    # h_e_slide = openslide.OpenSlide(os.path.join('slides_and_thumbs', '21-3263_1_1_e.mrxs'))
    # mpp_x, mpp_y, x_origin, y_origin = get_mpp_origin(h_e_slide)
    # tile_coords = np.array([13312, 50432])
    # slide_patch = h_e_slide.read_region((13312 * 4, 50432 * 4), 2,
    #                                     (SLIDE_PATCH_SIZE, SLIDE_PATCH_SIZE))
    # slide_patch.show()

    # h_e_slide = openslide.OpenSlide(os.path.join('slides_to_amit', '20-10015_1_1_e.mrxs'))
    # h_e_slide = openslide.OpenSlide(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '17-8750_2_10_a.mrxs'))
    # h_e_slide = openslide.OpenSlide(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '21-163_3_9_d.mrxs'))
    # h_e_slide = openslide.OpenSlide(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '21-2644_1_1_e.mrxs'))

    # h_e_slide = openslide.open_slide(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '21-3263_1_1_e.mrxs'))

    # thumb = Image.open(os.path.join('slides_to_amit', '0081_0_thumb_20-10015_1_1_e.jpg'))
    # thumb = Image.open(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '0030_0_thumb_17-8750_2_10_a.jpg'))
    # thumb = Image.open(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '0053_0_thumb_21-163_3_9_d.jpg'))
    # thumb = Image.open(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '0173_0_thumb_21-2644_1_1_e.png'))

    # qupath_location = (5500, 28750)  # 20-10015 slides
    # qupath_location = (5500, 28000)  # 20-10015 slides
    # qupath_location = (5000, 26500)  # 20-10015 slides

    # qupath_location = (4300, 29100)  # 17-8750 slides
    # qupath_location = (4150, 29000)  # 17-8750 slides
    # qupath_location = (2975, 29270)  # 17-8750 slides
    # qupath_location = (4700, 31000)  # 17-8750 slides
    # qupath_location = (5200, 33700)  # 17-8750 slides
    # qupath_location = (7850, 36100)  # 17-8750 slides
    # qupath_location = (11400, 30700)  # 17-8750 slides
    # qupath_location = (12230, 30880)  # 17-8750 slides

    # qupath_location = (16000, 10900)  # 21-163 slides
    # qupath_location = (3900, 12250)  # 21-163 slides
    # qupath_location = (8650, 6800)  # 21-163 slides
    # qupath_location = (7500, 16700)  # 21-163 slides
    # qupath_location = (6700, 16650)  # 21-163 slides

    # qupath_location = (7400, 30800)  # 21-2644 slides

    # qupath_location = (13800, 33500)  # 21-3263 slides
    # qupath_location = (10200, 41400)  # 21-3263 slides

    # width = int(h_e_slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_WIDTH))
    # height = int(h_e_slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_HEIGHT))

    # x_end, y_end = h_e_slide.level_dimensions[0]
    # # The following should hold:
    # x_end_validation = x_origin + width
    # y_end_validation = y_origin + height

    # rotation_img = cv2.imread(os.path.join('slides_to_amit', 'map_HE_20-10015_1_1_e_to_Her2_20-10015_1_1_m.png'), cv2.IMREAD_UNCHANGED)
    # rotation_img = cv2.imread(os.path.join('slides_to_amit2', '17-8750_2_10', 'map_HE_17-8750_2_10_a_to_Her2_17-8750_2_10_d.png'), cv2.IMREAD_UNCHANGED)
    # rotation_img = cv2.imread(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', 'map_HE_17-8750_2_10_a_labeled_to_Her2_17-8750_2_10_d_labeled.png'), cv2.IMREAD_UNCHANGED)
    # rotation_img = cv2.imread(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', 'map_HE_17-8750_2_10_a_to_Her2_17-8750_2_10_d.png'), cv2.IMREAD_UNCHANGED)
    # rotation_img = cv2.imread(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', 'map_HE_21-163_3_9_d_to_Her2_21-163_3_9_f.png'), cv2.IMREAD_UNCHANGED)
    # rotation_img = cv2.imread(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', 'map_HE_21-2644_1_1_e_to_Her2_21-2644_1_1_m.png'), cv2.IMREAD_UNCHANGED)

    # ihc_slide = openslide.OpenSlide(os.path.join('slides_to_amit', '20-10015_1_1_m.mrxs'))
    # ihc_slide = openslide.OpenSlide(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '17-8750_2_10_d.mrxs'))
    # ihc_slide = openslide.OpenSlide(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '21-163_3_9_f.mrxs'))
    # ihc_slide = openslide.OpenSlide(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '21-2644_1_1_m.mrxs'))

    # ihc_thumb = Image.open(os.path.join('slides_to_amit', '0004_0_thumb_20-10015_1_1_m.jpg'))
    # ihc_thumb = Image.open(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '0000_0_thumb_17-8750_2_10_d.jpg'))
    # ihc_thumb = Image.open(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '0028_0_thumb_21-163_3_9_f.jpg'))
    # ihc_thumb = Image.open(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '0093_0_thumb_21-2644_1_1_m.png'))
