import torch
import glob
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import cv2

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / (union + 1e-6)

def evaluate_predictions(model, image_dir, label_dir, num_images=1000, iou_threshold=0.5, conf_threshold=0.25):
    """
    Evaluate model predictions with additional debugging information
    """
    # Set model parameters
    model.conf = conf_threshold
    model.eval()
    
    # Initialize metrics
    class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    total_predictions = 0
    total_ground_truths = 0
    correct_classifications = 0
    
    # Get image files and limit to specified number
    image_files = glob.glob(os.path.join(image_dir, '*.png'))
    if not image_files:
        print(f"ERROR: No PNG images found in {image_dir}")
        print(f"Available files: {os.listdir(image_dir)}")
        return
        
    image_files = image_files[:num_images]
    
    print(f"\nEvaluating {len(image_files)} images...")
    
    # Debug: Print first few predictions
    debug_count = 0
    
    for idx, image_file in enumerate(tqdm(image_files, desc="Processing images")):
        # Debug information
        if debug_count < 5:
            print(f"\nProcessing image: {image_file}")
            
        # Get corresponding label file
        base_name = os.path.basename(image_file)
        label_file = os.path.join(label_dir, base_name.replace('.png', '.txt'))
        
        if not os.path.exists(label_file):
            print(f"Warning: No label file found for {base_name}")
            continue
            
        # Load and verify image
        img = cv2.imread(image_file)
        if img is None:
            print(f"Error: Could not load image {image_file}")
            continue
            
        actual_height, actual_width = img.shape[:2]
        
        # Load ground truth
        gt_boxes = []
        gt_classes = []
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Warning: Invalid label format in {label_file}")
                        continue
                        
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    
                    # Convert normalized coordinates to pixel coordinates
                    x1 = (x_center - width/2) * actual_width
                    y1 = (y_center - height/2) * actual_height
                    x2 = (x_center + width/2) * actual_width
                    y2 = (y_center + height/2) * actual_height
                    
                    gt_boxes.append([x1, y1, x2, y2])
                    gt_classes.append(class_id)
        except Exception as e:
            print(f"Error reading label file {label_file}: {str(e)}")
            continue
        
        # Debug ground truth
        if debug_count < 5:
            print(f"Ground truth boxes: {len(gt_boxes)}")
            print(f"Ground truth classes: {gt_classes}")
        
        # Run inference
        try:
            results = model(img)
            predictions = results.pred[0]
            
            # Debug predictions
            if debug_count < 5:
                print(f"Raw predictions shape: {predictions.shape}")
                print(f"Number of predictions: {len(predictions)}")
                if len(predictions) > 0:
                    print(f"First prediction: {predictions[0]}")
        except Exception as e:
            print(f"Error during inference on {image_file}: {str(e)}")
            continue
        
        # Process predictions
        pred_boxes = []
        pred_classes = []
        pred_scores = []
        
        if len(predictions) > 0:
            for pred in predictions:
                pred_boxes.append(pred[:4].cpu().numpy())
                pred_classes.append(int(pred[5].cpu().numpy()))
                pred_scores.append(float(pred[4].cpu().numpy()))
        
        # Debug processed predictions
        if debug_count < 5:
            print(f"Processed predictions:")
            print(f"Boxes: {pred_boxes}")
            print(f"Classes: {pred_classes}")
            print(f"Scores: {pred_scores}")
            debug_count += 1
        
        # Match predictions to ground truth
        matched_gt = set()
        
        for i, (pred_box, pred_class, pred_score) in enumerate(zip(pred_boxes, pred_classes, pred_scores)):
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth box
            for j, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                if j in matched_gt:
                    continue
                    
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            # Check if match is good enough
            if best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                if pred_class == gt_classes[best_gt_idx]:
                    correct_classifications += 1
                    class_metrics[pred_class]['tp'] += 1
                else:
                    class_metrics[pred_class]['fp'] += 1
            else:
                class_metrics[pred_class]['fp'] += 1
        
        # Count false negatives
        for gt_class in gt_classes:
            if len(matched_gt) < len(gt_classes):
                class_metrics[gt_class]['fn'] += 1
        
        total_predictions += len(pred_boxes)
        total_ground_truths += len(gt_boxes)
    
    # Print final metrics
    print("\n=== Final Evaluation Results ===")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total predictions: {total_predictions}")
    print(f"Total ground truths: {total_ground_truths}")
    
    if total_predictions == 0:
        print("\nWARNING: No predictions were made by the model!")
        print("Possible issues:")
        print("1. Confidence threshold too high")
        print("2. Model not properly loaded")
        print("3. Input image format/preprocessing issues")
        return
    
    # Overall metrics
    precision = correct_classifications / (total_predictions + 1e-6)
    recall = correct_classifications / (total_ground_truths + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    print(f"\nOverall Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    for class_id in sorted(class_metrics.keys()):
        metrics = class_metrics[class_id]
        class_precision = metrics['tp'] / (metrics['tp'] + metrics['fp'] + 1e-6)
        class_recall = metrics['tp'] / (metrics['tp'] + metrics['fn'] + 1e-6)
        class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall + 1e-6)
        
        print(f"\nClass {class_id}:")
        print(f"  Precision: {class_precision:.4f}")
        print(f"  Recall: {class_recall:.4f}")
        print(f"  F1 Score: {class_f1:.4f}")
        print(f"  True Positives: {metrics['tp']}")
        print(f"  False Positives: {metrics['fp']}")
        print(f"  False Negatives: {metrics['fn']}")

def main():
    # Load model
    print("Loading model...")
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', 
                              path='yolov5/runs/train/exp3/weights/best.pt', 
                              force_reload=True)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Set paths - update these to match your actual paths
    image_dir = 'yolo_dataset/images/train'  # Update this path
    label_dir = 'yolo_dataset/labels/train'  # Update this path
    
    # Verify directories exist
    if not os.path.exists(image_dir):
        print(f"Error: Image directory {image_dir} does not exist")
        return
    if not os.path.exists(label_dir):
        print(f"Error: Label directory {label_dir} does not exist")
        return
    
    # Run evaluation
    evaluate_predictions(model, 
                        image_dir, 
                        label_dir, 
                        num_images=1000,
                        iou_threshold=0.5, 
                        conf_threshold=0.25)

if __name__ == "__main__":
    main()