# YOLOv11x Inference Reproducibility Guide  
**Team:** *where is player18*  
**Detector Module for STC-2025 Submission*

---

## 📌 Overview

We used a fine-tuned version of **YOLOv11x** trained on pseudo-labels all from the STC-2025 dataset.
The model was further fine-tuned based on the STC-provided fine-tuned weights, using pseudo-labels we generated from the official training videos.

## ▶️ Inference Details

We applied **multi-scale test-time augmentation (TTA)** and custom **per-class NMS** using the following script:

```bash
python inference.py
```

Key inference parameters:

- **Input video**: `128057.mp4`
- **Output detection file**: `128057.txt` (MOT format)
- **Model weights**: `ultralytics-main\ultralytics-main\finetune_result\weights\finetune_best.pt`
- **Scales used**: `1280` and `1920`
- **NMS IoU threshold**: `0.5`
- **Confidence threshold**: `0.6`
- **TTA enabled**: `augment=True`

---

## 📄 Output Format

The result is saved in MOTChallenge format, with each line structured as:

```
<frame_id>, -1, <x>, <y>, <w>, <h>, <confidence>, -1, -1, -1
```

Example:

```
25, -1, 512.00, 190.00, 48.00, 110.00, 0.87, -1, -1, -1
```

This file is used as input for Deep-EIoU tracking.

---

## 🎬 Visualization

During inference, annotated frames are also saved into a video showing detection boxes and confidence scores.

---


