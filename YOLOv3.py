import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
from PIL import Image
import tqdm
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
from Detection import YOLOv3,decode_yolo_output,build_targets
from loss import YOLOLoss
from dataset import VOCDataset
                

def non_max_suppression(predictions, conf_threshold=0.5, iou_threshold=0.45):
    """
    Performs Non-Maximum Suppression (NMS) on inference results
    
    Args:
        predictions: tensor with shape [x1, y1, x2, y2, obj_conf, class_conf, class_pred]
        conf_threshold: objectness confidence threshold
        iou_threshold: IoU threshold for NMS
        
    Returns:
        List of detections with (x1, y1, x2, y2, object_conf, class_conf, class_pred)
    """
    # Filter out boxes with low object confidence
    mask = predictions[..., 4] > conf_threshold
    predictions = predictions[mask]
    
    # If none remain, return empty list
    if not predictions.size(0):
        return []
    
    # Get class with highest score
    class_scores, class_ids = torch.max(predictions[:, 5:], 1)
    
    # Create detection tensor
    detections = torch.cat([
        predictions[:, :4],  # boxes (x1, y1, x2, y2)
        predictions[:, 4].unsqueeze(1),  # objectness
        class_scores.unsqueeze(1),  # class score
        class_ids.float().unsqueeze(1)  # class id
    ], dim=1)
    
    # Run NMS for each class
    keep_boxes = []
    unique_labels = detections[:, 6].unique()
    
    for c in unique_labels:
        # Get detections with this class
        class_detections = detections[detections[:, 6] == c]
        
        # Sort by confidence
        _, conf_sort_idx = torch.sort(class_detections[:, 5], descending=True)
        class_detections = class_detections[conf_sort_idx]
        
        # Do NMS
        max_detections = []
        while class_detections.size(0):
            # Add the detection with highest confidence
            max_detections.append(class_detections[0].unsqueeze(0))
            
            # Stop if we have no more detections
            if len(class_detections) == 1:
                break
                
            # Get IoUs for all remaining boxes
            ious = box_iou(max_detections[-1][:, :4], class_detections[1:, :4])
            
            # Remove detections with IoU >= threshold
            class_detections = class_detections[1:][ious < iou_threshold]
        
        # Combine for this class
        if max_detections:
            keep_boxes.extend(max_detections)
    
    # Combine all kept boxes
    if keep_boxes:
        keep_boxes = torch.cat(keep_boxes).detach().cpu().numpy()
    else:
        keep_boxes = np.array([])
    
    return keep_boxes

def box_iou(box1, box2):
    """
    Compute IoU between two sets of boxes
    box1: [N, 4] where each row is [x1, y1, x2, y2]
    box2: [M, 4]
    Returns IoU: [N, M] where IoU[i,j] is the IoU between box1[i] and box2[j]
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    # Calculate the intersection area
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    
    # Apply max(0, x) to ensure no negative width/height
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate intersection area
    intersection = inter_w * inter_h
    
    # Calculate union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union = b1_area.unsqueeze(1) + b2_area - intersection
    
    # Calculate IoU
    iou = intersection / union
    
    return iou

def evaluate(model, dataloader, device, img_size, num_classes, conf_threshold=0.5, nms_threshold=0.45):
    """
    Evaluate the model on the validation set
    """
    model.eval()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for imgs, labels, img_ids in tqdm(dataloader, desc="Evaluating"):
            # Stack images into a single batch tensor
            imgs = torch.stack(imgs).to(device)
            
            # Forward pass
            outputs = model(imgs)
            
            # Process outputs
            batch_detections = []
            
            # Process each scale output
            for i, (pred, anchors) in enumerate(zip(outputs, model.anchors)):
                # Decode predictions
                boxes, objectness, class_scores = decode_yolo_output(
                    pred, anchors, img_size, num_classes
                )
                
                # Combine predictions from this scale
                batch_size, num_anchors, grid_size, _, _ = boxes.shape
                
                # Reshape everything for processing
                boxes = boxes.view(batch_size, -1, 4)  # [batch, grid*grid*num_anchors, 4]
                objectness = objectness.view(batch_size, -1, 1)  # [batch, grid*grid*num_anchors, 1]
                class_scores = class_scores.view(batch_size, -1, num_classes)  # [batch, grid*grid*num_anchors, classes]
                
                # Create detection tensor [batch, grid*grid*num_anchors, 5+num_classes]
                detections = torch.cat(
                    (boxes, objectness, class_scores), dim=2
                )
                
                batch_detections.append(detections)
            
            # Concatenate detections from all scales
            batch_detections = torch.cat(batch_detections, dim=1)
            
            # Process each image in batch
            for i in range(len(imgs)):
                image_preds = batch_detections[i]
                
                # Get detections above threshold
                score_mask = image_preds[:, 4] > conf_threshold
                detections = image_preds[score_mask]
                
                if len(detections) > 0:
                    # Apply NMS
                    nms_out = non_max_suppression(
                        detections.unsqueeze(0), 
                        conf_threshold, 
                        nms_threshold
                    )
                    
                    if len(nms_out) > 0:
                        for *box, obj_conf, cls_conf, cls_id in nms_out:
                            predictions.append([
                                img_ids[i],  # Image ID
                                *box,  # Box coordinates
                                obj_conf,  # Objectness confidence
                                cls_conf,  # Class confidence
                                cls_id  # Class ID
                            ])
                
                # Add ground truth
                img_targets = labels[i]
                
                # Filter out empty rows
                img_targets = img_targets[img_targets.sum(dim=1) > 0]
                
                if len(img_targets) > 0:
                    for box in img_targets:
                        cls_id, x_center, y_center, width, height = box
                        
                        # Convert to corner format
                        x1 = (x_center - width/2) * img_size
                        y1 = (y_center - height/2) * img_size
                        x2 = (x_center + width/2) * img_size
                        y2 = (y_center + height/2) * img_size
                        
                        targets.append([
                            img_ids[i],  # Image ID
                            cls_id.item(),  # Class ID
                            x1.item(),  # x1
                            y1.item(),  # y1
                            x2.item(),  # x2
                            y2.item()  # y2
                        ])
    
    # Convert to numpy arrays for easier processing
    predictions = np.array(predictions, dtype=object)
    targets = np.array(targets, dtype=object)
    
    # Calculate metrics
    # (This would typically involve mAP calculation)
    # Here we can implement evaluation based on VOC metrics
    # For simplicity, we'll just return the predictions and targets
    return predictions, targets
            
            

def calculate_map(predictions, targets, num_classes, iou_threshold=0.5):
    """
    Calculate mean Average Precision
    """
    # Sort predictions by confidence
    predictions = sorted(predictions, key=lambda x: x[5], reverse=True)
    
    # Initialize metrics
    average_precisions = []
    epsilon = 1e-6
    
    # Process each class
    for c in range(num_classes):
        # Get predictions and targets for this class
        class_preds = [p for p in predictions if p[6] == c]
        class_targets = [t for t in targets if t[1] == c]
        
        # Skip if no targets for this class
        if len(class_targets) == 0:
            continue
        
        # Count number of targets for this class
        num_targets = len(class_targets)
        
        # Initialize detected targets
        detected = [False] * len(class_targets)
        
        # Initialize true positives and false positives
        TP = np.zeros(len(class_preds))
        FP = np.zeros(len(class_preds))
        
        # Process each prediction
        for i, pred in enumerate(class_preds):
            pred_img_id = pred[0]
            pred_box = pred[1:5]
            
            # Find targets in same image with same class
            img_targets = [t for t in class_targets if t[0] == pred_img_id]
            
            best_iou = 0
            best_target_idx = -1
            
            # Find the target with highest IoU
            for j, target in enumerate(img_targets):
                target_box = target[2:6]
                
                # Calculate IoU
                iou = calculate_box_iou(pred_box, target_box)
                
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_target_idx = j
            
            # If we found a target
            if best_target_idx >= 0:
                # Check if this target was already detected
                target_idx = class_targets.index(img_targets[best_target_idx])
                if not detected[target_idx]:
                    # True positive
                    TP[i] = 1
                    detected[target_idx] = True
                else:
                    # False positive (already matched target)
                    FP[i] = 1
            else:
                # False positive (no matching target)
                FP[i] = 1
        
        # Compute cumulative false positives and true positives
        cum_FP = np.cumsum(FP)
        cum_TP = np.cumsum(TP)
        
        # Compute recall and precision
        recall = cum_TP / (num_targets + epsilon)
        precision = cum_TP / (cum_TP + cum_FP + epsilon)
        
        # Compute average precision (VOC method)
        ap = calculate_ap_voc(recall, precision)
        average_precisions.append(ap)
    
    # Return mAP
    mAP = sum(average_precisions) / len(average_precisions) if average_precisions else 0
    return mAP, average_precisions

def calculate_box_iou(box1, box2):
    """
    Calculate IoU between two boxes
    box1, box2: [x1, y1, x2, y2]
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    
    # Calculate intersection area
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    # Apply max(0, x) to ensure no negative width/height
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    
    # Calculate intersection area
    intersection = inter_w * inter_h
    
    # Calculate union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union = b1_area + b2_area - intersection
    
    # Calculate IoU
    iou = intersection / max(union, 1e-6)
    
    return iou

def calculate_ap_voc(recall, precision):
    """
    Calculate Average Precision according to the VOC 2007 11-point metric
    """
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.0
    return ap

def train(model, train_loader, val_loader, device, num_classes, img_size, 
          num_epochs=100, lr=0.001, weight_decay=0.0005, save_dir="checkpoints"):
    """
    Training function
    """
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler (reduce on plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # Loss function
    criterion = YOLOLoss()
    
    # Make sure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        # Progress bar for training
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, targets, _) in enumerate(train_bar):
            # Move to device - handle list of tensors
            images = torch.stack(images).to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Build target tensors
            target_tensors = build_targets(model, targets, img_size)
            target_tensors = [t.to(device) for t in target_tensors]
            
            # Calculate loss
            loss = criterion(outputs, target_tensors)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            
            # Update progress bar
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        # Average training loss
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, targets, _ in tqdm(val_loader, desc="Validation"):
                # Move to device - handle list of tensors
                images = torch.stack(images).to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Build target tensors
                target_tensors = build_targets(model, targets, img_size)
                target_tensors = [t.to(device) for t in target_tensors]
                
                # Calculate loss
                loss = criterion(outputs, target_tensors)
                
                # Update statistics
                val_loss += loss.item()
        
        # Average validation loss
        val_loss /= len(val_loader)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'yolov3_best.pth'))
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'yolov3_epoch_{epoch+1}.pth'))
  
                
   

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set hyperparameters
    img_size = 416  # Input image size
    batch_size = 8  # Batch size
    num_classes = 20  # VOC has 20 classes
    num_epochs = 100
    learning_rate = 0.001
    weight_decay = 0.0005
    
    # Paths
    data_root = "/home/prate/.cache/kagglehub/datasets/gopalbhattrai/pascal-voc-2012-dataset/versions/1"  # Replace with your VOC dataset path
    save_dir = "checkpoints"
    
    # Create model
    model = YOLOv3(num_classes=num_classes)
    model = model.to(device)
    
    # Print model summary
    print(model)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = VOCDataset(
        root_dir=data_root,
        split='train_val',
        img_size=img_size,
        transform=transform
    )
    
    val_dataset = VOCDataset(
        root_dir=data_root,
        split='test',
        img_size=img_size,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))  # Custom collate function for variable sized targets
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Train the model
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=num_classes,
        img_size=img_size,
        num_epochs=num_epochs,
        lr=learning_rate,
        weight_decay=weight_decay,
        save_dir=save_dir
    )
    
    # Evaluate the model
    print("Evaluating best model...")
    checkpoint = torch.load(os.path.join(save_dir, 'yolov3_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    predictions, targets = evaluate(
        model=model,
        dataloader=val_loader,
        device=device,
        img_size=img_size,
        num_classes=num_classes
    )
    
    # Calculate mAP
    mAP, average_precisions = calculate_map(
        predictions=predictions,
        targets=targets,
        num_classes=num_classes
    )
    
    print(f"mAP@0.5: {mAP:.4f}")
    
    # Print AP for each class
    for class_id, ap in enumerate(average_precisions):
        class_name = train_dataset.class_names[class_id]
        print(f"  {class_name}: {ap:.4f}")

if __name__ == "__main__":
    import kagglehub

    path = kagglehub.dataset_download("gopalbhattrai/pascal-voc-2012-dataset")

    print("Path to dataset files:", path)
    main()