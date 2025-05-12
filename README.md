# YOLOv3 From Scratch 🔍🧠

This repository contains a full implementation of the YOLOv3 (You Only Look Once) object detection algorithm written **from scratch** in Python using **PyTorch**. The goal is to deeply understand the architecture and internal workings of YOLOv3 without relying on pretrained weights or high-level APIs.

---

## 🚀 Features

- Clean and modular PyTorch implementation
- Custom model architecture (Darknet-53 backbone)
- Anchor boxes and bounding box predictions
- Multi-scale detection (3 detection heads)
- Custom dataloader for Pascal VOC-format datasets
- Full training and inference pipeline
- GPU support (CUDA)

---

## 🧠 Architecture

YOLOv3 uses a **Darknet-53** feature extractor followed by three detection layers operating at different scales. This implementation includes:

- Residual blocks for feature learning
- Upsampling and concatenation layers for multi-scale detection
- Bounding box prediction heads
- Sigmoid and objectness score processing
