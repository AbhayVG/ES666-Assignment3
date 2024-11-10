import os
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt

def load_images_from_folder(folder):
    print("Loading images from folder...")
    images = []
    for filename in sorted(os.listdir(folder)):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(np.array(img))
    print(f"Loaded {len(images)} images.")
    return images

def detect_and_match_features(img1, img2):
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append((kp1[m.queryIdx], kp2[m.trainIdx]))
    return good_matches

def calculate_transformation(matches):
    if len(matches) < 4:
        return None
    src_pts = np.float32([m[0].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([m[1].pt for m in matches]).reshape(-1, 2)

    # Simple transformation based on the average offset
    avg_offset = np.mean(dst_pts - src_pts, axis=0)
    transform_matrix = np.array([[1, 0, avg_offset[0]],
                                  [0, 1, avg_offset[1]],
                                  [0, 0, 1]])
    return transform_matrix

def warp_image(img, H, canvas_shape):
    h, w = canvas_shape[:2]
    warped_img = np.zeros(canvas_shape, dtype=img.dtype)

    y_coords, x_coords = np.indices((h, w))
    homogenous_coords = np.stack((x_coords.ravel(), y_coords.ravel(), np.ones_like(x_coords).ravel()))

    dest_coords = H @ homogenous_coords
    dest_coords /= dest_coords[2, :]
    dest_coords = dest_coords[:2, :].astype(int)

    valid_coords = (
        (0 <= dest_coords[0]) & (dest_coords[0] < img.shape[1]) &
        (0 <= dest_coords[1]) & (dest_coords[1] < img.shape[0])
    )

    warped_img[y_coords.ravel()[valid_coords], x_coords.ravel()[valid_coords]] = \
        img[dest_coords[1][valid_coords], dest_coords[0][valid_coords]]

    print("Image warped.")
    return warped_img

def stitch_images(image_folder):
    images = load_images_from_folder(image_folder)
    base_img = images[0]
    canvas_height, canvas_width = base_img.shape[0], base_img.shape[1] * len(images)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    offset_x = 0
    canvas[:, offset_x:offset_x + base_img.shape[1]] = base_img
    offset_x += base_img.shape[1]

    cumulative_H = np.eye(3)

    for i in range(1, len(images)):
        img = images[i]

        matches = detect_and_match_features(base_img, img)

        H = calculate_transformation(matches)
        if H is None:
            continue

        cumulative_H = np.dot(cumulative_H, H)

        warped_img = warp_image(img, cumulative_H, canvas.shape)

        mask = (warped_img > 0).any(axis=2)
        canvas[mask] = warped_img[mask]

        offset_x += img.shape[1]
        base_img = img


    return canvas

# List of folders to process
folders = [
    '/content/drive/MyDrive/ES666CV/ES666-Assignment3/Images/I1',
    '/content/drive/MyDrive/ES666CV/ES666-Assignment3/Images/I2',
    '/content/drive/MyDrive/ES666CV/ES666-Assignment3/Images/I3',
    '/content/drive/MyDrive/ES666CV/ES666-Assignment3/Images/I4',
    '/content/drive/MyDrive/ES666CV/ES666-Assignment3/Images/I5',
    '/content/drive/MyDrive/ES666CV/ES666-Assignment3/Images/I6'
]

# Process each folder and plot results
for folder in folders:
    result = stitch_images(folder)
    plt.figure(figsize=(40, 30))
    plt.imshow(result.astype(np.uint8))
    plt.axis("off")
    plt.title(f"Stitched Image from {folder}")
    plt.show()
