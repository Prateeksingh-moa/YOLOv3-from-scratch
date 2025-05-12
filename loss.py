import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        # Use binary cross entropy with logits for objectness and class predictions
        self.bce_obj = nn.BCEWithLogitsLoss()
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        
        # Loss coefficients
        self.lambda_coord = 5
        self.lambda_obj = 1
        self.lambda_noobj = 0.5
        self.lambda_class = 1
    
    def forward(self, predictions, targets):
        """
        predictions: list of 3 tensors [small_output, medium_output, large_output]
        targets: list of 3 tensors [small_targets, medium_targets, large_targets]
        """
        total_loss = 0
        
        # Process each scale separately
        for scale_idx, (pred, target) in enumerate(zip(predictions, targets)):
            batch_size = pred.shape[0]
            num_anchors = 3
            grid_size = pred.shape[2]
            num_classes = target.shape[-1] - 5
            
            # Reshape prediction to match target shape [batch, 3*(5+num_classes), grid, grid]
            # to [batch, grid, grid, 3, 5+num_classes]
            pred = pred.view(batch_size, num_anchors, 5 + num_classes, grid_size, grid_size)
            pred = pred.permute(0, 3, 4, 1, 2).contiguous()
            
            # Object mask - grid cells with objects
            obj_mask = target[..., 4] > 0.5
            noobj_mask = target[..., 4] <= 0.5
            
            # Check if we have any objects (prevents division by zero)
            if obj_mask.sum() > 0:
                # Calculate object loss using BCE with logits
                obj_loss = self.bce_obj(
                    pred[..., 4][obj_mask],
                    target[..., 4][obj_mask]
                )
                
                # Calculate loss for x, y coordinates (using sigmoid targets)
                # Get xy predictions and targets
                xy_pred = torch.sigmoid(pred[..., :2][obj_mask.unsqueeze(-1).expand_as(pred[..., :2])])
                xy_target = target[..., :2][obj_mask.unsqueeze(-1).expand_as(target[..., :2])]
                
                # Calculate XY loss
                xy_loss = self.mse(xy_pred, xy_target)
                
                # Calculate loss for width and height (log scale)
                # Get wh predictions and targets
                wh_pred = pred[..., 2:4][obj_mask.unsqueeze(-1).expand_as(pred[..., 2:4])]
                wh_target = target[..., 2:4][obj_mask.unsqueeze(-1).expand_as(target[..., 2:4])]
                
                # Apply clipping to prevent extreme values
                wh_pred = torch.clamp(wh_pred, -10.0, 10.0)
                
                # Calculate WH loss
                wh_loss = self.mse(wh_pred, wh_target)
                
                # Class loss with BCE logits
                class_loss = self.bce_cls(
                    pred[..., 5:][obj_mask.unsqueeze(-1).expand_as(pred[..., 5:])],
                    target[..., 5:][obj_mask.unsqueeze(-1).expand_as(target[..., 5:])]
                )
            else:
                # If no objects, zero these losses
                obj_loss = torch.tensor(0.0, device=pred.device)
                xy_loss = torch.tensor(0.0, device=pred.device)
                wh_loss = torch.tensor(0.0, device=pred.device)
                class_loss = torch.tensor(0.0, device=pred.device)
            
            # Calculate no object loss (lower weight for background)
            # Make sure we have background cells
            if noobj_mask.sum() > 0:
                noobj_loss = self.bce_obj(
                    pred[..., 4][noobj_mask],
                    target[..., 4][noobj_mask]
                )
            else:
                noobj_loss = torch.tensor(0.0, device=pred.device)
            
            # Check for NaN values before combining losses
            for loss_name, loss_value in [
                ("obj_loss", obj_loss),
                ("noobj_loss", noobj_loss),
                ("xy_loss", xy_loss),
                ("wh_loss", wh_loss),
                ("class_loss", class_loss)
            ]:
                if torch.isnan(loss_value).any() or torch.isinf(loss_value).any():
                    print(f"WARNING: {loss_name} contains NaN or Inf values")
                    # Replace with small value to prevent propagation
                    if loss_name == "wh_loss":  # Width-height is most likely to cause problems
                        loss_value = torch.tensor(0.1, device=pred.device)
                    else:
                        loss_value = torch.tensor(0.0, device=pred.device)
            
            # Combine all losses with their coefficients
            scale_loss = (
                self.lambda_coord * (xy_loss + wh_loss)
                + self.lambda_obj * obj_loss
                + self.lambda_noobj * noobj_loss
                + self.lambda_class * class_loss
            )
            
            # Check final scale_loss for NaN
            if torch.isnan(scale_loss).any() or torch.isinf(scale_loss).any():
                print(f"WARNING: Scale {scale_idx} loss contains NaN or Inf values - using default")
                scale_loss = torch.tensor(0.1, device=pred.device)  # Safe default value
            
            total_loss += scale_loss
        
        return total_loss