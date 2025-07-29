import cv2
import numpy as np

def compute_homography():
    # Manually picking 4 points on the front-view image
    src_pts = np.float32([
        [580, 460], # top-left
        [700, 460], # top-right
        [1120, 720],# bottom-right
        [200, 720] # bottom-left
    ])

    # Mapping them to a rectangle in BEV
    dst_pts = np.float32([
        [300,   0],
        [1000,  0],
        [1000, 720],
        [300,  720]
    ])
    return cv2.getPerspectiveTransform(src_pts, dst_pts)

def apply_homography(img, H, output_size=(1280, 720)):
    return cv2.warpPerspective(img, H, output_size)

if __name__ == "__main__":
    # local test
    img = cv2.imread("dataset/raw_rgb/rgb_000001.png")
    if img is None:
        print("Place a test RGB at 'dataset/raw_rgb/rgb_000001.png'")
        exit(1)
    H = compute_homography()
    bev = apply_homography(img, H)
    cv2.imshow("BEV", bev)
    cv2.waitKey(0)
