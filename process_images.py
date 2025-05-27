import os
import cv2
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


# Read CSV Annotations
df_annotations = pd.read_csv("./duhovky/iris_annotation.csv")

# Circle IOU + Evaluation
def circle_iou(c1, c2, eps=1e-9):
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    area1 = math.pi * (r1**2)
    area2 = math.pi * (r2**2)
    dx, dy = (x1 - x2), (y1 - y2)
    d = math.sqrt(dx*dx + dy*dy)
    
    # No overlap
    if d >= (r1 + r2):
        return 0.0
    # One circle completely inside the other
    if d <= abs(r2 - r1):
        inter = math.pi * (min(r1, r2)**2)
        union = area1 + area2 - inter
        return inter / (union + eps)
    
    # Partial overlap => lens area
    r1_sq, r2_sq = r1*r1, r2*r2
    alpha = math.acos((d*d + r1_sq - r2_sq)/(2*d*r1))*2
    beta  = math.acos((d*d + r2_sq - r1_sq)/(2*d*r2))*2
    area1_seg = 0.5*r1_sq*(alpha - math.sin(alpha))
    area2_seg = 0.5*r2_sq*(beta - math.sin(beta))
    inter = area1_seg + area2_seg
    union = area1 + area2 - inter
    
    return inter / (union + eps)

def evaluate_detection(gt_circles, pred_circles, iou_threshold=0.75):
    used_pred = set()
    TP = 0
    
    for gt in gt_circles:
        best_iou = 0.0
        best_idx = -1
        
        for i, pred in enumerate(pred_circles):
            if i in used_pred:
                continue
            iou_val = circle_iou(gt, pred)
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = i
        if best_iou >= iou_threshold:
            TP += 1
            used_pred.add(best_idx)
            
    FP = len(pred_circles) - len(used_pred)
    FN = len(gt_circles) - TP
    
    return TP, FP, FN

def precision_recall_f1(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    
    return precision, recall, f1

# Detection
def detect_circles(image, dp, min_dist, param1, param2, min_radius, max_radius, center=True):
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    result = []
    if circles is not None:
        circles = np.around(circles).astype(int)
        for c in circles[0, :]:
            x, y, r = c
            if center and (140 < x < 180 and 140 < y < 180):
                result.append((x, y, r))
            elif not center and y > 200:
                result.append((x, y, r))
                
    return result

def detect_lower_eyelid(image, dp, min_dist, param1, param2, min_radius, max_radius, ceter=True):
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    result = []
    if circles is not None:
        circles = np.around(circles).astype(int)
        for c in circles[0, :]:
            x, y, r = c
            # hard coded condition for lower eyelid detection
            # keep circles in [110..210, 110..210]
            if (110 < x < 210) and (110 < y < 210):
                result.append((x, y, r))
                
    return result

if __name__ == "__main__":
    # lil grid search 
    # for pupil / iris:
    dp_values = [1.2, 1.5]         
    min_radius_values = [20, 30, 50]  

    # for eyelids:
    eyelid_min_dist_values = [150, 200]  
    eyelid_dp_values = [1.2, 1.5]       

    # loop over each image  
    all_global_TP = 0
    all_global_FP = 0
    all_global_FN = 0

    best_params_per_image = []  # details

    print(len(df_annotations))

    for idx, row in df_annotations.iterrows():
        image_relpath = row["image"]  # 4xmpl "190/L/S1190L02.jpg"
        full_path = os.path.join("./zadanie1/duhovky", image_relpath) 
        print(idx)
        # Ground truth circles
        gt_circles = [
            (int(row["center_x_1"]), int(row["center_y_1"]), int(row["polomer_1"])),
            (int(row["center_x_2"]), int(row["center_y_2"]), int(row["polomer_2"])),
            (int(row["center_x_3"]), int(row["center_y_3"]), int(row["polomer_3"])),
            (int(row["center_x_4"]), int(row["center_y_4"]), int(row["polomer_4"]))
        ]
        
        # Load & preprocess image
        if not os.path.exists(full_path):
            # missing image
            continue
        
        image_gray = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            continue  # bad read
        
        # Pad to 320x320
        h, w = image_gray.shape
        pad_h = 320 - h if (320 - h) > 0 else 0 
        image_padded = cv2.copyMakeBorder(
            image_gray, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=0
        )
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_clahe = clahe.apply(image_padded)
        
        image_denoised = cv2.fastNlMeansDenoising(
            image_clahe, h=10, templateWindowSize=7, searchWindowSize=21
        )
        
        # all combinations for pupil & iris & eyelids -> pick best F1
        best_f1_for_image = 0.0
        best_params_for_image = None
        best_pred_circles_for_image = None

        for dp_pupil in dp_values:
            for r_pupil in min_radius_values:
                for dp_iris in dp_values:
                    for r_iris in min_radius_values:
                        for dp_eyelid in eyelid_dp_values:
                            for dist_eyelid in eyelid_min_dist_values:
                                # Pupils
                                pupil_circles = detect_circles(
                                    image_denoised,
                                    dp=dp_pupil, min_dist=100,
                                    param1=100, param2=20,
                                    min_radius=r_pupil, max_radius=50,
                                    center=True
                                )
                                # Iris
                                iris_circles = detect_circles(
                                    image_denoised,
                                    dp=dp_iris, min_dist=100,
                                    param1=100, param2=35,
                                    min_radius=r_iris, max_radius=100,
                                    center=True
                                )
                                # Upper eyelid
                                upper_eyelid_circles = detect_circles(
                                    image_denoised,
                                    dp=dp_eyelid, min_dist=dist_eyelid,
                                    param1=100, param2=30,
                                    min_radius=160, max_radius=300,
                                    center=False
                                )
                                # Lower eyelid
                                lower_eyelid_circles = detect_lower_eyelid(
                                    image_denoised,
                                    dp=dp_eyelid, min_dist=dist_eyelid,
                                    param1=100, param2=30,
                                    min_radius=160, max_radius=200
                                )
                                
                                # Combine
                                predicted_circles = (pupil_circles + iris_circles
                                                    + upper_eyelid_circles + lower_eyelid_circles)
                                
                                TP, FP, FN = evaluate_detection(gt_circles, predicted_circles, 0.75)
                                prec, rec, f1 = precision_recall_f1(TP, FP, FN)
                                
                                if f1 > best_f1_for_image:
                                    best_f1_for_image = f1
                                    best_params_for_image = {
                                        "dp_pupil": dp_pupil,
                                        "r_pupil": r_pupil,
                                        "dp_iris": dp_iris,
                                        "r_iris": r_iris,
                                        "dp_eyelid": dp_eyelid,
                                        "dist_eyelid": dist_eyelid,
                                        "TP": TP, "FP": FP, "FN": FN,
                                        "precision": prec,
                                        "recall": rec,
                                        "f1": f1
                                    }
                                    best_pred_circles_for_image = predicted_circles[:]
        
        # save the best combo
        if best_params_for_image is not None:
            best_params_per_image.append({
                "image": image_relpath,
                **best_params_for_image
            })
            
            # Accumulating global TP,FP,FN with the best set
            all_global_TP += best_params_for_image["TP"]
            all_global_FP += best_params_for_image["FP"]
            all_global_FN += best_params_for_image["FN"]


    # Final overall precision/recall/f1
    global_precision, global_recall, global_f1 = precision_recall_f1(
        all_global_TP, all_global_FP, all_global_FN
    )

    print(f"{'-'*30}")
    print("Finished for all images")
    print(f"Global TP={all_global_TP}, FP={all_global_FP}, FN={all_global_FN}")
    print(f"Global Precision={global_precision:.3f}, Recall={global_recall:.3f}, F1={global_f1:.3f}")
    print(f"{'-'*30}")

    # CSV of best params:
    df_best = pd.DataFrame(best_params_per_image)
    df_best.to_csv("best_params_per_image.csv", index=False)
    print("Saved best_params_per_image.csv!")