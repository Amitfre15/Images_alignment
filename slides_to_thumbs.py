import os
import math
import numpy as np
from PIL import Image
import cv2

OPENSLIDE_PATH = r"C:\Program Files\openslide-bin-4.0.0.3-windows-x64\openslide-bin-4.0.0.3-windows-x64\bin"

if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def slide_to_thumb_coord(slide, thumb, slide_coord):
    """
    Converts a coordinate from the slide to the corresponding coordinate in the thumbnail.

    Parameters:
    - slide: The OpenSlide object.
    - thumb: The thumbnail image (PIL Image or similar).
    - slide_coord: A tuple (x, y) representing the coordinate on the slide.

    Returns:
    - A tuple (x, y) representing the corresponding coordinate on the thumbnail.
    """
    # Get dimensions of the slide
    slide_width, slide_height = slide.dimensions

    # Get dimensions of the thumbnail
    thumb_width, thumb_height = thumb.size

    # Calculate scale factors
    scale_x = thumb_width / slide_width
    scale_y = thumb_height / slide_height

    # Convert slide coordinates to thumbnail coordinates
    thumb_x = int(slide_coord[0] * scale_x)
    thumb_y = int(slide_coord[1] * scale_y)

    return thumb_x, thumb_y


def thumb_to_slide_coord(thumb, slide, thumb_coord):
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
    thumb_width, thumb_height = thumb.size

    # Get dimensions of the slide
    slide_width, slide_height = slide.dimensions

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


def extract_and_show_patch(slide, thumb, rotation_mat, slide_coords=None, thumb_coords=None, prefix=''):
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
        # Calculate the original patch width and height
        patch_width = slide_coords[2] - slide_coords[0]
        patch_height = slide_coords[3] - slide_coords[1]

        # Calculate the padding using sqrt(2) scaling factor
        # scaling_factor = math.sqrt(2)
        scaling_factor = 2
        padding_width = int((scaling_factor - 1) * patch_width / 2)
        padding_height = int((scaling_factor - 1) * patch_height / 2)

        # Add padding to the coordinates
        padded_slide_coords = (
            max(0, slide_coords[0] - padding_width),
            max(0, slide_coords[1] - padding_height),
            min(slide.dimensions[0], slide_coords[2] + padding_width),
            min(slide.dimensions[1], slide_coords[3] + padding_height)
        )

        # Map the slide coordinates to thumbnail coordinates
        thumb_coord1 = slide_to_thumb_coord(slide, thumb, (slide_coords[0], slide_coords[1]))
        thumb_coord2 = slide_to_thumb_coord(slide, thumb, (slide_coords[2], slide_coords[3]))
        thumb_coords = list(thumb_coord1) + list(thumb_coord2)
        rotation_angle = (rotation_mat[thumb_coords[0], thumb_coords[1]][-1] / 100) - 90

        # Extract padded slide patch
        padded_slide_width = padded_slide_coords[2] - padded_slide_coords[0]
        padded_slide_height = padded_slide_coords[3] - padded_slide_coords[1]
        slide_patch = slide.read_region((padded_slide_coords[0], padded_slide_coords[1]), 0,
                                        (padded_slide_width, padded_slide_height))
        actual_pad_width = (slide_patch.width - patch_width) // 2
        actual_pad_height = (slide_patch.height - patch_height) // 2
        corners = [actual_pad_width, actual_pad_height,
                   slide_patch.width - actual_pad_width, actual_pad_height,
                   actual_pad_width, slide_patch.height - actual_pad_height,
                   slide_patch.width - actual_pad_width, slide_patch.height - actual_pad_height]
        slide_patch = slide_patch.rotate(rotation_angle, expand=True)

        # Crop the padded patch
        rotated_corners = rotate_coordinates(corners=corners, angle=rotation_angle)
        new_center = slide_patch.width // 2, slide_patch.height // 2
        xs = [rc[0] + new_center[0] for rc in rotated_corners]
        ys = [rc[1] + new_center[1] for rc in rotated_corners]
        small_x, big_x = min(xs), max(xs)
        small_y, big_y = min(ys), max(ys)
        crop_box_slide = (small_x, small_y, big_x, big_y)
        slide_patch = slide_patch.crop(crop_box_slide)
    else:
        # Map the thumbnail coordinates to slide coordinates
        slide_coord1 = thumb_to_slide_coord(slide=slide, thumb=thumb, thumb_coord=(thumb_coords[0], thumb_coords[1]))
        slide_coord2 = thumb_to_slide_coord(slide=slide, thumb=thumb, thumb_coord=(thumb_coords[2], thumb_coords[3]))
        slide_coords = list(slide_coord1) + list(slide_coord2)
        # rotation_angle = rotation_mat[thumb_coords[0], thumb_coords[1]][-1] / 100
        slide_patch = slide.read_region((slide_coords[0], slide_coords[1]), 0,
                                        (slide_coords[2] - slide_coords[0], slide_coords[3] - slide_coords[1]))

    # Extract thumb patches
    thumb_patch = thumb.crop((thumb_coords[0], thumb_coords[1], thumb_coords[2], thumb_coords[3]))

    # Convert slide patch to RGB (it might be RGBA, depending on the format)
    slide_patch_rgb = slide_patch.convert("RGB")
    # slide_patch_rgb.show(title="Slide Patch")
    # thumb_patch = thumb_patch.rotate(rotation_angle, expand=True)

    # if prefix == 'ihc_':
    # Display both patches
    thumb_patch.show(title="Thumbnail Patch")
    slide_patch_rgb.show(title="Slide Patch")
    slide_patch_rgb.save(fp=f'{prefix}slide_patch.png', format="PNG")
    thumb_patch.save(fp=f'{prefix}thumb_patch.png', format="PNG")

    return thumb_coords


def main():
    h_e_slide = openslide.OpenSlide(os.path.join('slides_to_amit', '20-10015_1_1_e.mrxs'))
    thumb = Image.open(os.path.join('slides_to_amit', '0081_0_thumb_20-10015_1_1_e.jpg'))

    x_origin = int(h_e_slide.properties.get('openslide.bounds-x'))
    y_origin = int(h_e_slide.properties.get('openslide.bounds-y'))
    mpp_x = float(h_e_slide.properties.get('openslide.mpp-x'))
    mpp_y = float(h_e_slide.properties.get('openslide.mpp-y'))

    # qupath_location = (5500, 28750)
    qupath_location = (5500, 28000)
    openslide_location = (round(qupath_location[0] / mpp_x) + x_origin, round(qupath_location[1] / mpp_y) + y_origin)

    width = int(h_e_slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_WIDTH))
    height = int(h_e_slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_HEIGHT))

    x_end, y_end = h_e_slide.level_dimensions[0]
    # The following should hold:
    x_end_validation = x_origin + width
    y_end_validation = y_origin + height

    slide_coords = tuple(list(openslide_location) + [cor + 8000 for cor in openslide_location])
    # rotation_img = Image.open(os.path.join('slides_to_amit', 'map_HE_20-10015_1_1_e_to_Her2_20-10015_1_1_m.png'))
    rotation_img = cv2.imread(os.path.join('slides_to_amit', 'map_HE_20-10015_1_1_e_to_Her2_20-10015_1_1_m.png'), cv2.IMREAD_UNCHANGED)
    rotation_img = cv2.cvtColor(rotation_img, cv2.COLOR_BGR2RGB)
    rotation_mat = np.array(rotation_img).transpose(1, 0, 2)

    thumb_coords = extract_and_show_patch(h_e_slide, thumb, rotation_mat, slide_coords=slide_coords)
    thumb_center = [(thumb_coords[0] + thumb_coords[2]) // 2, (thumb_coords[1] + thumb_coords[3]) // 2]
    thumb_width, thumb_height = thumb_coords[2] - thumb_coords[0], thumb_coords[3] - thumb_coords[1]
    # corresp_ihc_thumb_coords = rotation_mat[[thumb_coords[0], thumb_coords[2]], [thumb_coords[1], thumb_coords[3]]]
    # corresp_ihc_thumb_coords = [cor for x_y_cor in corresp_ihc_thumb_coords[:, :2] for cor in x_y_cor]
    ihc_center = list(rotation_mat[thumb_center[0], thumb_center[1]][:2])

    # IHC
    ihc_slide = openslide.OpenSlide(os.path.join('slides_to_amit', '20-10015_1_1_m.mrxs'))
    ihc_thumb = Image.open(os.path.join('slides_to_amit', '0004_0_thumb_20-10015_1_1_m.jpg'))
    x_ratio = ihc_thumb.width / thumb.width
    y_ratio = ihc_thumb.height / thumb.height
    corresp_ihc_thumb_coords = [max(0, ihc_center[0] - (thumb_width // 2) * x_ratio),
                                max(0, ihc_center[1] - (thumb_height // 2) * y_ratio),
                                min(ihc_thumb.height, ihc_center[0] + (thumb_width // 2) * x_ratio),  # rotation_mat is rotated by 90 deg
                                min(ihc_thumb.width, ihc_center[1] + (thumb_height // 2) * y_ratio)]  # rotation_mat is rotated by 90 deg
    # rotation_mat is rotated by 90 deg
    corresp_ihc_thumb_coords = [ihc_thumb.width - corresp_ihc_thumb_coords[i + 1] if i % 2 == 0 else corresp_ihc_thumb_coords[i - 1] for i, coord in
                                enumerate(corresp_ihc_thumb_coords)]
    corresp_ihc_thumb_coords[0], corresp_ihc_thumb_coords[2] = corresp_ihc_thumb_coords[2], corresp_ihc_thumb_coords[0]
    # ihc_thumb_coords = [coord // x_ratio if i % 2 == 0 else coord // y_ratio for i, coord in enumerate(corresp_ihc_thumb_coords)]
    extract_and_show_patch(ihc_slide, ihc_thumb, rotation_mat, thumb_coords=corresp_ihc_thumb_coords, prefix='ihc_')


if __name__ == '__main__':
    main()
