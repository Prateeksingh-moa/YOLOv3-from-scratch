import torch
import torch.nn as nn
from Darknet import ResidualBlock, Darknet53

class YOLODetectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(YOLODetectionBlock, self).__init__()

        self.features = nn.Sequential(
            # 1x1 convolution to reduce channels (BottleNeck design)
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),

            # 3x3 convolution to learn spatial patterns
            nn.Conv2d(out_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        
        # Final conv maps to output prediction tensor - we keep this separate
        # to avoid applying batch norm and activation
        self.output = nn.Conv2d(in_channels, 3 * (5 + num_classes), 1, 1, 0)
    
    def forward(self, x):
        x = self.features(x)
        return self.output(x)

class YOLOv3(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv3, self).__init__()
        
        # Anchors for each scale (small, medium, large objects)
        # These values should be adjusted based on your dataset (width, height)
        self.anchors = [
            [(10, 13), (16, 30), (33, 23)],      # Small objects
            [(30, 61), (62, 45), (59, 119)],     # Medium objects
            [(116, 90), (156, 198), (373, 326)]  # Large objects
        ]
        
        self.num_classes = num_classes
        
        # Backbone: Darknet-53
        self.darknet = Darknet53()
        
        # Upsampling layers for feature fusion
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
        # Additional convolutional layers for feature map processing
        # For large objects (Layer 5 output)
        self.conv_large_1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1)
        )
        
        # For medium objects (after upsampling and concatenation)
        self.conv_medium_1 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, kernel_size=1, stride=1, padding=0, bias=False),  # Fixed: 512 from large + 512 from route_medium
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )
                
        # For small objects (after upsampling and concatenation)
        self.conv_small_1 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=1, stride=1, padding=0, bias=False),  # Fixed: 256 from medium + 256 from route_small
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        
        # Detection blocks for three scales
        self.detect_large = YOLODetectionBlock(512, 256, num_classes)
        self.detect_medium = YOLODetectionBlock(256, 128, num_classes)
        self.detect_small = YOLODetectionBlock(128, 64, num_classes)
    
    def forward(self, x):
        # Get features from Darknet backbone
        features = self.darknet(x)
        
        # We need outputs from layers 3, 4, 5 for multi-scale detection
        route_small = features[3]   # Layer 3 output (256 channels)
        route_medium = features[4]  # Layer 4 output (512 channels)
        route_large = features[5]   # Layer 5 output (1024 channels)
        
        # Large object detection branch
        large_feat = self.conv_large_1(route_large)
        large_output = self.detect_large(large_feat)
        
        # Upsampling and concatenation for medium objects
        up_large = self.upsample(large_feat)
        concat_medium = torch.cat([up_large, route_medium], dim=1)  # 512 + 512 = 1024 channels
        medium_feat = self.conv_medium_1(concat_medium)
        medium_output = self.detect_medium(medium_feat)
        
        # Upsampling and concatenation for small objects
        up_medium = self.upsample(medium_feat)
        concat_small = torch.cat([up_medium, route_small], dim=1)  # 256 + 256 = 512 channels
        small_feat = self.conv_small_1(concat_small)
        small_output = self.detect_small(small_feat)
        
        return [small_output, medium_output, large_output]
  
      

def decode_yolo_output(pred, anchors, img_dim, num_classes):
    """
    Decodes the network's output tensor into bounding box predictions
    
    Args:
        pred: Raw network output tensor [batch, 3*(5+num_classes), grid_size, grid_size]
        anchors: Anchor boxes for this grid scale, shape [3, 2]
        img_dim: Image dimensions (assumed square)
        num_classes: Number of classes
        
    Returns:
        boxes: Predicted bounding boxes [x1, y1, x2, y2] (in pixel coords)
        objectness: Objectness score
        class_scores: Class probability scores
    """
    batch_size = pred.shape[0]
    grid_size = pred.shape[2]
    stride = img_dim // grid_size
    bbox_attrs = 5 + num_classes
    num_anchors = 3
    
    # Reshape from [batch, 3*(5+num_classes), grid_size, grid_size]
    # to [batch, 3, grid_size, grid_size, 5+num_classes]
    pred = pred.view(batch_size, num_anchors, bbox_attrs, grid_size, grid_size)
    pred = pred.permute(0, 1, 3, 4, 2).contiguous()
    
    # Apply sigmoid to the objectness score and class probabilities
    objectness = torch.sigmoid(pred[..., 4])
    class_scores = torch.sigmoid(pred[..., 5:])
    
    # Create grid offsets
    grid_x = torch.arange(grid_size).repeat(grid_size, 1).view(
        [1, 1, grid_size, grid_size]).type_as(pred)
    grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view(
        [1, 1, grid_size, grid_size]).type_as(pred)
    
    # Shape anchors for the grid
    anchors_tensor = torch.FloatTensor(anchors).to(pred.device)
    anchors_tensor = anchors_tensor.view(1, num_anchors, 1, 1, 2)
    
    # Apply formulas to get box coordinates
    # bx = sigmoid(tx) + cx, by = sigmoid(ty) + cy
    # bw = pw * exp(tw), bh = ph * exp(th)
    box_xy = torch.sigmoid(pred[..., :2]) + torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], dim=4)
    box_wh = torch.exp(pred[..., 2:4]) * anchors_tensor
    
    # Scale back to image size
    box_xy = box_xy * stride
    box_wh = box_wh * stride
    
    # Convert from center coordinates to corner coordinates
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    
    # Combine to get final boxes [x1, y1, x2, y2]
    boxes = torch.cat([box_x1y1, box_x2y2], dim=4)
    
    return boxes, objectness, class_scores

def build_targets(model, targets_list, img_size):
    """
    Build target tensors for each scale from ground truth boxes
    
    Args:
        model: YOLOv3 model
        targets_list: List of ground truth targets tensors, one per image
        img_size: Input image size
    
    Returns:
        List of target tensors for each scale [small_targets, medium_targets, large_targets]
    """
    batch_size = len(targets_list)
    device = next((t.device for t in targets_list if len(t) > 0), torch.device('cpu'))
    num_classes = model.num_classes
    
    # Initialize target tensors for each detection scale
    small_grid_size = img_size // 8   # 52 for 416x416 input
    medium_grid_size = img_size // 16  # 26 for 416x416 input
    large_grid_size = img_size // 32  # 13 for 416x416 input
    
    targets_small = torch.zeros((batch_size, small_grid_size, small_grid_size, 3, 5 + num_classes), device=device)
    targets_medium = torch.zeros((batch_size, medium_grid_size, medium_grid_size, 3, 5 + num_classes), device=device)
    targets_large = torch.zeros((batch_size, large_grid_size, large_grid_size, 3, 5 + num_classes), device=device)
    
    # Scale strides
    strides = [img_size // small_grid_size, img_size // medium_grid_size, img_size // large_grid_size]
    
    # Get the model anchors
    anchors_small = torch.tensor(model.anchors[0], device=device)
    anchors_medium = torch.tensor(model.anchors[1], device=device) 
    anchors_large = torch.tensor(model.anchors[2], device=device)
    
    all_anchors = [anchors_small, anchors_medium, anchors_large]
    all_targets = [targets_small, targets_medium, targets_large]
    grid_sizes = [small_grid_size, medium_grid_size, large_grid_size]
    
    # Process each image's targets
    for i, targets in enumerate(targets_list):
        # Skip empty targets
        if targets.shape[0] == 0:
            continue
        
        # Process each ground truth box
        for box in targets:
            # Skip invalid boxes or handle NaNs
            if torch.isnan(box).any() or torch.isinf(box).any():
                continue
                
            class_id, x_center, y_center, width, height = box
            
            # Convert to Python float/int for validation
            class_id_int = int(class_id.item())
            x_center = float(x_center.item())
            y_center = float(y_center.item())
            width = float(width.item())
            height = float(height.item())
            
            # Skip invalid boxes
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and width > 0 and height > 0):
                continue
                
            # Check if class_id is valid (between 0 and num_classes-1)
            if not (0 <= class_id_int < num_classes):
                continue
            
            # Find the best matching anchor and scale for this box
            best_scale_idx = -1
            best_anchor_idx = -1
            best_iou = -1
            
            # Convert width and height from normalized to pixel values
            box_w = width * img_size
            box_h = height * img_size
            box_area = box_w * box_h
            
            # Find best anchor across all scales
            for scale_idx in range(3):
                anchors = all_anchors[scale_idx]
                
                for anchor_idx, anchor in enumerate(anchors):
                    anchor_w, anchor_h = anchor
                    
                    # Calculate IoU between box and anchor
                    min_w = min(box_w, anchor_w)
                    min_h = min(box_h, anchor_h)
                    
                    intersection = min_w * min_h
                    union = box_area + (anchor_w * anchor_h) - intersection
                    
                    if union > 0:
                        iou = intersection / union
                        if iou > best_iou:
                            best_iou = iou
                            best_scale_idx = scale_idx
                            best_anchor_idx = anchor_idx
            
            if best_scale_idx == -1:
                # If no good anchor is found, choose the first one as fallback
                best_scale_idx = 0
                best_anchor_idx = 0
            
            # Get the grid size for the best scale
            grid_size = grid_sizes[best_scale_idx]
            
            # Calculate grid cell coordinates
            # Convert normalized coordinates to grid coordinates
            grid_x = int(x_center * grid_size)
            grid_y = int(y_center * grid_size)
            
            # Clamp to valid grid indices
            grid_x = max(0, min(grid_x, grid_size - 1))
            grid_y = max(0, min(grid_y, grid_size - 1))
            
            # Calculate the exact position within the grid cell
            x_offset = x_center * grid_size - grid_x
            y_offset = y_center * grid_size - grid_y
            
            # Get the target tensor for this scale
            target_tensor = all_targets[best_scale_idx]
            
            # Assign target values - ensure indices are valid
            if (i < batch_size and 
                0 <= grid_y < grid_size and 
                0 <= grid_x < grid_size and 
                0 <= best_anchor_idx < 3 and 
                0 <= class_id_int < num_classes):
                
                # Assign coordinates - we store the x/y offsets which should be in [0,1]
                target_tensor[i, grid_y, grid_x, best_anchor_idx, 0] = x_offset  # x offset
                target_tensor[i, grid_y, grid_x, best_anchor_idx, 1] = y_offset  # y offset
                
                # For width and height, we store log values to match the network output
                # Convert width and height to log scale relative to anchors
                stride = strides[best_scale_idx]
                anchor_w, anchor_h = all_anchors[best_scale_idx][best_anchor_idx]
                target_tensor[i, grid_y, grid_x, best_anchor_idx, 2] = torch.log(width * img_size / anchor_w + 1e-6)
                target_tensor[i, grid_y, grid_x, best_anchor_idx, 3] = torch.log(height * img_size / anchor_h + 1e-6)
                
                # Objectness
                target_tensor[i, grid_y, grid_x, best_anchor_idx, 4] = 1.0  
                
                # Class (one-hot encoding)
                target_tensor[i, grid_y, grid_x, best_anchor_idx, 5 + class_id_int] = 1.0
    
    # Return in order: [small_output, medium_output, large_output]
    return [targets_small, targets_medium, targets_large]