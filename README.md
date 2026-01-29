# GTATrack-STC2025

**Execution-level Reproduction of a Global Tracklet Association Pipeline for Soccer Multi-Object Tracking**

This repository provides an **execution-level reproduction** of the tracking pipeline used in the  
**STC-2025 Soccer Multi-Object Tracking** submission.

The pipeline combines strong detection, online tracking, and offline global identity refinement,  
and is designed for **long-term, identity-consistent tracking** under heavy occlusion and fisheye distortion.

> YOLOv11x (Detection) → Deep-EIoU (Online MOT) → GTA-Link (Offline Global Association)

This repository focuses on **execution-level reproducibility rather than architectural novelty**.

---

## 📌 Pipeline Overview

The complete processing pipeline consists of the following stages:

1. **Detection**  
   YOLOv11x inference with multi-scale test-time augmentation

2. **Online Tracking**  
   Deep-EIoU using motion cues and ReID appearance features

3. **Offline Refinement**  
   GTA-Link for global tracklet association (tracklet splitting and reconnection)

4. **Output**  
   Standard MOT-format tracking results

---

## ⚡ Environment Setup

```bash
conda create -n gtat python=3.10 -y
conda activate gtat
pip install -r requirements.txt
```

---

## 🔁 Reproducibility

This repository preserves the original execution workflow used in the STC-2025 submission.
Exact commands, parameters, and execution notes for each stage are documented as follows:

### Execution Records

- **YOLOv11x detection**  
  `YOLOV11X_REPRODUCE.txt`

- **Deep-EIoU online tracking**  
  `DEEP_EIOU_REPRODUCE.txt`

- **GTA-Link offline refinement**  
  `GTA_LINK_REPRODUCE.txt`

- **End-to-end pipeline description**  
  `STC2025_Pipeline.txt`

All commands match those used in the original submission.

---

## 📂 Repository Layout (Execution-level Reproduction)
The following directory structure reflects the actual repository organization.  
This project intentionally avoids refactoring external codebases into a unified framework.

```bash
GTATrack-STC2025/
├── Deep-EIoU/                          # official Deep-EIoU repository (unmodified)
├── gta-link/                           # official GTA-Link repository (unmodified)
├── ultralytics-main/                   # YOLOv11x inference code
├── DEEP_EIOU_REPRODUCE.txt              # Deep-EIoU commands and parameters
├── GTA_LINK_REPRODUCE.txt               # GTA-Link refinement commands
├── YOLOV11X_REPRODUCE.txt               # YOLOv11x inference details
├── STC2025_Pipeline.txt                 # overall pipeline description
├── requirements.txt
└── README.md

```
Intermediate artifacts such as extracted frames, detection outputs, and tracking results
are generated at runtime and are therefore not committed to this repository.

---

## 🔗 External Dependencies

This project relies on the following official implementations, used without structural modification:

- **YOLOv11x (Ultralytics)**  
  https://github.com/ultralytics/ultralytics

- **Deep-EIoU**  
  https://github.com/MCG-NJU/Deep-EIoU

- **GTA-Link**  
  https://github.com/sjc042/gta-link

---

## 📄 Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{jian2025gtatrack,
  title     = {GTATrack: Winner Solution to SoccerTrack 2025 with Deep-EIoU and Global Tracklet Association},
  author    = {Jian, Rong-Lin and Luo, Ming-Chi and Huang, Chen-Wei and Lee, Chia-Ming and Lin, Yu-Fan and Hsu, Chih-Chung},
  booktitle = {Proceedings of the 8th International ACM Workshop on Multimedia Content Analysis in Sports (MMSports '25)},
  year      = {2025},
  pages     = {180--188},
  publisher = {ACM},
  doi       = {10.1145/3728423.3759416}
}
```
---

## 🙏 Acknowledgements
This repository builds upon outstanding open-source work in object detection,
multi-object tracking, and global tracklet association.
