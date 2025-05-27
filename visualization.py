import cv2
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from process_images import detect_circles, detect_lower_eyelid


if __name__ == "__main__":

    df_annotations = pd.read_csv("./duhovky/iris_annotation.csv")

    # single image
    image_relpath = "190/L/S1190L02.jpg"
    full_path = "./duhovky/190/L/S1190L02.jpg"

    row = df_annotations[df_annotations["image"] == image_relpath]
    if row.empty:
        raise ValueError(f"No annotation found for image {image_relpath}!")

    # Ground-truth circles
    gt_circles = [
        (int(row["center_x_1"]), int(row["center_y_1"]), int(row["polomer_1"])),
        (int(row["center_x_2"]), int(row["center_y_2"]), int(row["polomer_2"])),
        (int(row["center_x_3"]), int(row["center_y_3"]), int(row["polomer_3"])),
        (int(row["center_x_4"]), int(row["center_y_4"]), int(row["polomer_4"])),
    ]

    # Load & Preprocess (Pad => 320x320, CLAHE, Denoising)
    image_gray = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    height, width = image_gray.shape
    padding_bottom = 320 - height
    image_padded = cv2.copyMakeBorder(
        image_gray, 0, padding_bottom, 0, 0,
        cv2.BORDER_CONSTANT, value=0
    )

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image_padded)

    image_denoised = cv2.fastNlMeansDenoising(
        image_clahe, h=10, templateWindowSize=7, searchWindowSize=21
    )

    # Detections
    pupil_circles = detect_circles(
        image_denoised, dp=1.5, min_dist=100, param1=100, param2=20,
        min_radius=25, max_radius=50
    )
    iris_circles = detect_circles(
        image_denoised, dp=1.5, min_dist=100, param1=100, param2=35,
        min_radius=50, max_radius=100
    )
    upper_eyelid_circles = detect_circles(
        image_denoised, dp=1.2, min_dist=200, param1=100, param2=30,
        min_radius=160, max_radius=300, center=False
    )
    lower_eyelid_circles = detect_lower_eyelid(
        image_denoised, dp=1.5, min_dist=100, param1=100, param2=30,
        min_radius=160, max_radius=200
    )

    predicted_circles = (
        pupil_circles
        + iris_circles
        + upper_eyelid_circles
        + lower_eyelid_circles
    )

    # Visualization of the detections 
    output_img = cv2.cvtColor(image_padded.copy(), cv2.COLOR_GRAY2BGR)

    # lower eyelid (cyan)
    for (x, y, r) in pupil_circles:
        cv2.circle(output_img, (x, y), r, (0, 0, 255), 2)
    # iris (green)
    for (x, y, r) in iris_circles:
        cv2.circle(output_img, (x, y), r, (0, 255, 0), 2)
    # upper eyelid (blue)
    for (x, y, r) in upper_eyelid_circles:
        cv2.circle(output_img, (x, y), r, (255, 0, 0), 2)
    # pupils (red)
    for (x, y, r) in lower_eyelid_circles:
        cv2.circle(output_img, (x, y), r, (255, 255, 0), 2)

    # ground-truth circles (white)
    for (gx, gy, rg) in gt_circles:
        cv2.circle(output_img, (gx, gy), rg, (255, 255, 255), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Circles (Pupil=Red, Iris=Green, Eyelids=Blue/Cyan), GT=White")
    plt.axis("off")
    plt.show()

    # (F) segmenting the the iris with mask
    if len(iris_circles) > 0:
        # largest circle should represent the iris so
        # picking the circle with the largest radius
        x_iris, y_iris, r_iris = max(iris_circles, key=lambda c: c[2])
        
        # Mask for the iris
        iris_mask = np.zeros(image_padded.shape[:2], dtype=np.uint8)
        
        cv2.circle(iris_mask, (x_iris, y_iris), r_iris, 1, thickness=-1)
        
        # Show the iris mask 
        plt.figure(figsize=(8, 8))
        plt.imshow(iris_mask, cmap="gray")
        plt.title("Binary Mask for Iris (1=inside, 0=outside)")
        plt.axis("off")
        plt.show()
        
        # Segmentation on the grayscale image:
        iris_segmented = image_padded.copy()
        iris_segmented[iris_mask == 0] = 0 
        
        plt.figure(figsize=(8, 8))
        plt.imshow(iris_segmented, cmap="gray")
        plt.title("Iris-Segmented Grayscale Image")
        plt.axis("off")
        plt.show()
        
    else:
        print("No iris circle found, cannot segment.")