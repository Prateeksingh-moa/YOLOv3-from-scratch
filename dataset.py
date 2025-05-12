import torch
from torch.utils.data import DataLoader,Dataset
import os
import xml.etree.ElementTree as ET
import numpy as np
import tqdm
from tqdm.auto import tqdm
from PIL import Image
import torchvision.transforms as transforms

class VOCDataset(Dataset):
    def __init__(self, root_dir, split='train_val', img_size=416, transform=None, preload_annotations=True):
        """
        Pascal VOC dataset loader with tqdm progress bars
        
        Args:
            root_dir: Root directory of VOC dataset
            split: 'train_val' or 'test'
            img_size: Input size for model
            transform: Image transformations
            preload_annotations: Whether to preload all annotations (faster but uses more memory)
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.preload_annotations = preload_annotations
        
        # Class names (20 classes for VOC)
        self.class_names = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", 
            "bus", "car", "cat", "chair", "cow", 
            "diningtable", "dog", "horse", "motorbike", "person", 
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
        
        # Check if directories exist based on the folder structure in the image
        if split == 'train_val':
            self.annotations_dir = os.path.join(root_dir, "VOC2012_train_val", "VOC2012_train_val", "Annotations")
            self.img_dir = os.path.join(root_dir, "VOC2012_train_val", "VOC2012_train_val", "JPEGImages")
        else:  # test split
            self.annotations_dir = os.path.join(root_dir, "VOC2012_test", "VOC2012_test", "Annotations")
            self.img_dir = os.path.join(root_dir, "VOC2012_test", "VOC2012_test", "JPEGImages")
        
        # Validate paths
        if not os.path.exists(self.annotations_dir):
            raise FileNotFoundError(f"Annotations directory not found: {self.annotations_dir}")
        
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Images directory not found: {self.img_dir}")
        
        # Get all XML annotation files
        print(f"Loading {split} dataset from {self.annotations_dir}")
        self.annotation_files = [f for f in os.listdir(self.annotations_dir) 
                               if f.endswith('.xml')]
        
        print(f"Found {len(self.annotation_files)} annotation files")
        
        # Preload annotations if requested
        self.cached_annotations = None
        if preload_annotations:
            self.cached_annotations = self.load_all_annotations()
    
    def load_all_annotations(self):
        """
        Pre-load all annotations with progress bar for initialization
        """
        annotations = []
        print("Preloading annotations...")
        for ann_file in tqdm(self.annotation_files, desc="Loading annotations"):
            xml_path = os.path.join(self.annotations_dir, ann_file)
            boxes, classes = self.parse_voc_xml(xml_path)
            annotations.append((boxes, classes))
        return annotations
    
    def __len__(self):
        return len(self.annotation_files)
    
    def parse_voc_xml(self, xml_file):
        """
        Parse VOC XML annotation file to extract bounding boxes and class ids
        """
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image size
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            boxes = []
            classes = []
            
            # Extract all object annotations
            for obj in root.findall('object'):
                # Get class name
                class_name = obj.find('name').text
                if class_name not in self.class_to_idx:
                    continue  # Skip classes not in our list
                    
                class_id = self.class_to_idx[class_name]
                
                # Check if the object is difficult
                difficult = obj.find('difficult')
                if difficult is not None and int(difficult.text) == 1:
                    # Skip difficult objects if you want
                    # continue
                    pass
                
                # Get bounding box
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text) / img_width
                ymin = float(bbox.find('ymin').text) / img_height
                xmax = float(bbox.find('xmax').text) / img_width
                ymax = float(bbox.find('ymax').text) / img_height
                
                # Ensure coordinates are valid
                xmin = max(0, min(1, xmin))
                ymin = max(0, min(1, ymin))
                xmax = max(0, min(1, xmax))
                ymax = max(0, min(1, ymax))
                
                # Skip invalid boxes
                if xmax <= xmin or ymax <= ymin:
                    print(f"Skipping invalid box in {os.path.basename(xml_file)}: "
                          f"({xmin}, {ymin}, {xmax}, {ymax})")
                    continue
                
                # Convert to YOLO format (center_x, center_y, width, height)
                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0
                width = xmax - xmin
                height = ymax - ymin
                
                boxes.append([x_center, y_center, width, height])
                classes.append(class_id)
            
            return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.float32)
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    
    def __getitem__(self, idx):
    # Get annotation file
        ann_file = self.annotation_files[idx]
        img_id = os.path.splitext(ann_file)[0]
    
        try:
            # Load image
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                # Try .jpeg extension
                img_path = os.path.join(self.img_dir, f"{img_id}.jpeg")
                if not os.path.exists(img_path):
                    # Try .png extension
                    img_path = os.path.join(self.img_dir, f"{img_id}.png")
            
            img = Image.open(img_path).convert("RGB")
            
            # Parse annotation (either from cache or from file)
            if self.cached_annotations is not None:
                boxes, classes = self.cached_annotations[idx]
            else:
                xml_path = os.path.join(self.annotations_dir, ann_file)
                boxes, classes = self.parse_voc_xml(xml_path)
            
            # Prepare image
            orig_width, orig_height = img.size
            
            # Resize image to model input size
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
            
            # Convert to tensor
            img = transforms.ToTensor()(img)
            
            if self.transform:
                img = self.transform(img)
            
            # Create target tensor with class and box info
            num_boxes = len(boxes)
            targets = torch.zeros((num_boxes, 5))
            
            if num_boxes > 0:
                # Format: [class_id, x_center, y_center, width, height]
                targets[:, 0] = torch.from_numpy(classes)
                targets[:, 1:] = torch.from_numpy(boxes)
            
            return img, targets, img_id
            
        except Exception as e:
            print(f"Error loading image or annotation for {img_id}: {e}")
            # Return a placeholder in case of error
            img = torch.zeros((3, self.img_size, self.img_size))
            targets = torch.zeros((0, 5))
            return img, targets, img_id