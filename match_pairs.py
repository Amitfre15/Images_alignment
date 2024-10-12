import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.morphology import disk
from skimage.morphology import binary_opening
from skimage.measure import regionprops, label
import open3d as o3d
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors


# Function to load and normalize image
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


# Initialize paths
# folder = '../5 samples/Marked/'
folder = os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2')
thumb_HE = '0030_0_thumb_17-8750_2_10_a_labeled.jpg'
thumb_Her2 = '0000_0_thumb_17-8750_2_10_d_labeled.jpg'
output_mapping_file = f'map_HE_{thumb_HE[13:-4]}_to_Her2_{thumb_Her2[13:-4]}.png'

# Load images
im_HE = load_image(os.path.join(folder, thumb_HE))
im_Her2 = load_image(os.path.join(folder, thumb_Her2))

# Display images (optional)
def display_image(img, title="Image"):
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.show()


display = True
if display:
    display_image(im_HE, "H&E Image")
    display_image(im_Her2, "Her2 Image")

# Find landmarks in H&E thumb
color_landmark = np.array([0.0156863, 0.2, 1])
# For each pixel, measure its distance from color_landmark
dist_HE = np.sqrt(np.sum((im_HE - color_landmark) ** 2, axis=2))

th = 0.03
dist_HE_b = dist_HE < th  # % distance < th should be true for the pixels of the landmarks
dist_HE_b = binary_opening(dist_HE_b, disk(1))  # remove small noise pixels

if display:
    display_image(dist_HE_b.astype(np.float64), "Landmarks H&E")

# Extract centroid of landmarks
labeled_HE = label(dist_HE_b)  # get all connected components
regions_HE = regionprops(labeled_HE)
S_land_HE = np.array([region.centroid for region in regions_HE])

# Repeat for Her2
dist_Her2 = np.sqrt(np.sum((im_Her2 - color_landmark) ** 2, axis=2))
dist_Her2_b = dist_Her2 < th
dist_Her2_b = binary_opening(dist_Her2_b, disk(1))

if display:
    display_image(dist_Her2_b.astype(np.float64), "Landmarks Her2")

labeled_Her2 = label(dist_Her2_b)
regions_Her2 = regionprops(labeled_Her2)
S_land_Her2 = np.array([region.centroid for region in regions_Her2])

# Ensure the number of landmarks is the same
if len(S_land_HE) != len(S_land_Her2):
    raise ValueError("Number of landmarks do not match.")


# Perform Iterative Closest Point (ICP) to find the correspondence between the landmarks
def run_icp(source, target, trans_init, threshold):
    """Runs ICP and returns the result."""
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    return reg_p2p.transformation, reg_p2p.fitness


# Visualize the results
def plot_landmarks(landmarks_fixed, landmarks_moving, title_fixed, title_moving):
    import matplotlib.pyplot as plt
    plt.scatter(landmarks_fixed[:, 0], landmarks_fixed[:, 1], c='b', label=title_fixed)
    plt.scatter(landmarks_moving[:, 0], landmarks_moving[:, 1], c='r', label=title_moving)
    plt.legend()
    plt.show()


def get_rotation_matrix_90_degrees(axis, angle_deg):
    """Returns a rotation matrix for 90-degree increments along a given axis."""
    angle_rad = np.radians(angle_deg)
    if axis == 'x':
        rotation_vector = [angle_rad, 0, 0]
    elif axis == 'y':
        rotation_vector = [0, angle_rad, 0]
    elif axis == 'z':
        rotation_vector = [0, 0, angle_rad]
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    return o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)


pts_moving = S_land_Her2
pts_fixed = S_land_HE / 2

S_land_HE_3d = np.hstack([pts_fixed, np.zeros((len(S_land_HE), 1))])  # For point cloud
S_land_Her2_3d = np.hstack([pts_moving, np.zeros((len(S_land_Her2), 1))])
plot_landmarks(S_land_HE_3d, S_land_Her2_3d, 'H&E Landmarks', 'Initial Her2 Landmarks')

pcd_moving = o3d.geometry.PointCloud()
pcd_fixed = o3d.geometry.PointCloud()
pcd_moving.points = o3d.utility.Vector3dVector(S_land_Her2_3d)
pcd_fixed.points = o3d.utility.Vector3dVector(S_land_HE_3d)
fixed_center = pcd_fixed.get_center()[:2]


# Define angles for 90-degree rotations
angles = [0, 90, 180, 270]

threshold = 500
trans_init = np.eye(4)

# List to store results for different rotations
results = []

# Perform ICP for 90-degree rotations (90°, 180°, 270°)
for i in range(0, 4):
    # Generate a 90-degree rotation matrix around the z-axis (change axis if needed)
    rotation_matrix = get_rotation_matrix_90_degrees('z', 90 * i)

    # Construct a 4x4 transformation matrix
    trans_rot = np.eye(4)
    trans_rot[:3, :3] = rotation_matrix

    # Multiply the rotation matrix with the initial transformation matrix
    trans_init_rotated = np.dot(trans_rot, trans_init)

    pcd_moving_tmp = o3d.geometry.PointCloud()
    rotated_points = np.dot(S_land_Her2_3d, trans_init_rotated[:3, :3].T)
    pcd_moving_tmp.points = o3d.utility.Vector3dVector(rotated_points)
    plot_landmarks(S_land_HE_3d, rotated_points, 'H&E Landmarks', 'Transformed Her2 Landmarks')

    translation = fixed_center - pcd_moving_tmp.get_center()[:2]
    trans_rot_points = rotated_points + np.hstack([translation, np.zeros(1)])
    pcd_moving_tmp.points = o3d.utility.Vector3dVector(trans_rot_points)
    plot_landmarks(S_land_HE_3d, trans_rot_points, 'H&E Landmarks', 'Transformed Her2 Landmarks')

    # Run ICP with the rotated transformation
    transformation_rotated, fitness_rotated = run_icp(pcd_moving_tmp, pcd_fixed, trans_init, threshold)

    # Store result
    results.append((transformation_rotated, fitness_rotated, trans_rot_points))

# Select the best transformation based on fitness
best_transformation, best_fitness, best_rot_points = max(results, key=lambda x: x[1])

# Output the best transformation
print("Best transformation is:")
print(best_transformation)
print("With fitness:", best_fitness)

# Perform transformation
S_land_Her2_transformed = np.dot(best_rot_points, best_transformation[:3, :3].T) + best_transformation[:3, 3]


if display:
    plot_landmarks(S_land_HE_3d, S_land_Her2_3d, 'H&E Landmarks', 'Initial Her2 Landmarks')
    plot_landmarks(S_land_HE_3d, S_land_Her2_transformed, 'H&E Landmarks', 'Transformed Her2 Landmarks')

# KNN search to map Her2 landmarks to H&E landmarks
knn = NearestNeighbors(n_neighbors=1)
knn.fit(S_land_Her2_transformed[:, :2])
# knn.fit(best_transformed_points[:, :2])
HE_Her2_land_mapping = knn.kneighbors(S_land_HE_3d[:, :2], return_distance=False).flatten()

# Mapping for H&E and Her2
PT_HE = S_land_HE
PT_Her2 = S_land_Her2[HE_Her2_land_mapping]

# Calculate the Euclidean distance between corresponding landmarks
pairs_dist = np.sqrt(np.sum((S_land_Her2_transformed[HE_Her2_land_mapping] - S_land_HE_3d)**2, axis=1))
# pairs_dist = np.sqrt(np.sum((best_transformed_points[HE_Her2_land_mapping] - S_land_HE_3d)**2, axis=1))

# Print the maximum distance between pairs
print(f'max pair dist: {np.max(pairs_dist):.3f}')

# Threshold for problematic distances
th_pixels = 100

# Check if any landmark pair exceeds the threshold
if np.max(pairs_dist) > th_pixels:
    print(f'There is at least one landmark pair with more than {th_pixels} distance')


# Delaunay triangulation
tri = Delaunay(PT_HE)

# Define bounding box and pixels in H&E
min_y, min_x = np.floor(np.min(PT_HE, axis=0)).astype(int)
max_y, max_x = np.ceil(np.max(PT_HE, axis=0)).astype(int)
y, x = np.meshgrid(np.arange(min_y, max_y), np.arange(min_x, max_x))
pixels = np.vstack([y.ravel(), x.ravel()]).T

# Triangle indices for H&E image
triangle_indices = tri.find_simplex(pixels)

# Initialize transformed image and mapping image
im_transformed = np.copy(im_HE)
im_map = np.zeros_like(im_HE)

# Create the (x, y) grid for the Her2 image
im_XY = np.dstack(np.meshgrid(np.arange(im_Her2.shape[1]), np.arange(im_Her2.shape[0])))

# Iterate over each triangle and compute the transformation
for i in range(len(tri.simplices)):
    # Get the pixels inside the current triangle in H&E
    tri_curr_pixels_HE = pixels[triangle_indices == i]

    if len(tri_curr_pixels_HE) == 0:
        continue

    # Get the vertices of the triangle in H&E and Her2
    tri_curr_PT_HE = PT_HE[tri.simplices[i]]
    tri_curr_PT_Her2 = PT_Her2[tri.simplices[i]]

    # Compute the affine transformation matrix between the two triangles
    A = np.linalg.lstsq(np.hstack([tri_curr_PT_HE, np.ones((3, 1))]), tri_curr_PT_Her2, rcond=None)[0]

    # Apply the transformation to the pixels of the current triangle
    tri_curr_pixels_HE_hom = np.hstack([tri_curr_pixels_HE, np.ones((len(tri_curr_pixels_HE), 1))])
    pixels_transformed_Her2 = np.dot(tri_curr_pixels_HE_hom, A)

    # Round and clip the coordinates for image indexing
    pixels_transformed_Her2 = np.round(pixels_transformed_Her2).astype(int)
    pixels_transformed_Her2 = np.clip(pixels_transformed_Her2, 0, np.array(im_Her2.shape[:2]) - 1)

    # Convert pixel coordinates to integer after rounding
    y_transformed = np.round(pixels_transformed_Her2[:, 0]).astype(int)
    x_transformed = np.round(pixels_transformed_Her2[:, 1]).astype(int)

    y_curr = np.round(tri_curr_pixels_HE[:, 0]).astype(int)
    x_curr = np.round(tri_curr_pixels_HE[:, 1]).astype(int)

    # Filter out any out-of-bound indices
    valid_mask = (x_transformed >= 0) & (x_transformed < im_Her2.shape[1]) & \
                 (y_transformed >= 0) & (y_transformed < im_Her2.shape[0]) & \
                 (x_curr >= 0) & (x_curr < im_HE.shape[1]) & \
                 (y_curr >= 0) & (y_curr < im_HE.shape[0])

    x_transformed = x_transformed[valid_mask]
    y_transformed = y_transformed[valid_mask]
    x_curr = x_curr[valid_mask]
    y_curr = y_curr[valid_mask]

    # Update the transformed image for each channel (R, G, B)
    im_transformed[y_curr, x_curr, 0] = im_Her2[y_transformed, x_transformed, 0]  # Red channel
    im_transformed[y_curr, x_curr, 1] = im_Her2[y_transformed, x_transformed, 1]  # Green channel
    im_transformed[y_curr, x_curr, 2] = im_Her2[y_transformed, x_transformed, 2]  # Blue channel
    display_image(im_transformed[min_y - 700:max_y + 700, :], "Transformed Her2 on H&E")

    im_map[y_curr, x_curr, 0] = im_XY[y_transformed, x_transformed, 0]
    im_map[y_curr, x_curr, 1] = im_XY[y_transformed, x_transformed, 1]

    # Compute the rotation (theta) and store in the blue channel
    svd_U, _, svd_V = np.linalg.svd(A[:2, :2])
    R_mat = np.dot(svd_U, svd_V)
    theta = np.arctan2(R_mat[1, 0], R_mat[0, 0])

    if theta < 0:
        theta += 2 * np.pi

    theta = (theta / (2 * np.pi)) * 360 * 100  # Convert to degrees and scale
    # im_map.ravel()[ind_curr_b] = theta
    im_map[y_curr, x_curr, 2] = theta

# Display the transformed image
if display:
    display_image(im_transformed[min_y-700:max_y+700, :], "Transformed Her2 on H&E")

# Save mapping result as a 16-bit PNG image
# im_map_uint16 = np.clip(im_map * 65535, 0, 65535).astype(np.uint16)  # unnecessary

im_map_uint16 = im_map.astype(np.uint16)
# im_map_uint16 = cv2.cvtColor(im_map, cv2.COLOR_RGB2BGR).astype(np.uint16)
cv2.imwrite(output_mapping_file, im_map_uint16)

# Verify the saved file
im_map_uint16_rec = cv2.imread(output_mapping_file, cv2.IMREAD_UNCHANGED)
if not np.array_equal(im_map_uint16, im_map_uint16_rec):
    print("Warning: The saved mapping file does not match the original.")
