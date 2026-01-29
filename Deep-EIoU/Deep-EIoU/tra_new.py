# -*- coding: utf-8 -*-
import os
import os.path as osp
import numpy as np
import cv2
import argparse
from loguru import logger
from tracker.Deep_EIoU import Deep_EIoU
from reid.torchreid.utils import FeatureExtractor
from tqdm import tqdm   # ✅ 加入 tqdm
from collections import deque

# ✅ 解決 libiomp5md.dll 重複初始化錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def make_parser():
    parser = argparse.ArgumentParser("DeepEIoU Tracking with MOT Format Detections")
    parser.add_argument("--proximity_thresh", type=float, default=0.4,
                        help="Distance threshold for proximity filtering")
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')
    parser.add_argument("--data-root", default="data/football", help="Path to data folder")
    parser.add_argument("--img-dir", default="img2", help="Directory containing frame images")
    parser.add_argument("--det-dir", default="det2", help="Directory containing MOT format detections")
    parser.add_argument("--output", default="tracking_results.txt", help="Output tracking result file")
    parser.add_argument("--reid-model", default="checkpoints/sports_model.pth.tar-60", help="ReID model path")
    parser.add_argument("--with-reid", action="store_true", default=True, help="Enable ReID feature matching")

    # Tracking parameters
    parser.add_argument("--track-high-thresh", type=float, default=0.7, help="Tracking confidence threshold")
    parser.add_argument("--track-low-thresh", type=float, default=0.4, help="Lowest detection threshold")
    parser.add_argument("--new-track-thresh", type=float, default=0.8, help="New track threshold")
    parser.add_argument("--track-buffer", type=int, default=90, help="Frames to keep lost tracks")
    parser.add_argument("--match-thresh", type=float, default=0.8, help="Matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=10, help="Filter out tiny boxes")
    
    # 新增遮擋處理參數
    parser.add_argument("--occ-iou-thresh", type=float, default=0.5,
                        help="IoU threshold for occlusion detection")
    parser.add_argument("--occ-sim-thresh", type=float, default=0.3,
                        help="Similarity threshold for occlusion detection")
    parser.add_argument("--occ-buffer-frames", type=int, default=20,
                        help="Number of frames to buffer occluded tracks")

    return parser

def load_mot_detections(det_dir):
    """Load MOT format detections from a single txt file."""
    detections = {}
    if not osp.exists(det_dir):
        logger.error(f"Detection file not found: {det_dir}")
        return detections

    det_data = np.loadtxt(det_dir, delimiter=",")
    if det_data.ndim == 1:
        det_data = det_data[np.newaxis, :]  # 只有一行時補一個維度

    for row in det_data:
        frame_id = int(row[0])
        x1, y1, w, h, conf = row[2], row[3], row[4], row[5], row[6]
        x2 = x1 + w
        y2 = y1 + h
        bbox = np.array([x1, y1, x2, y2, conf])
        if frame_id not in detections:
            detections[frame_id] = []
        detections[frame_id].append(bbox)

    # Convert list to ndarray per frame
    for k in detections:
        detections[k] = np.array(detections[k])
    return detections

def main(args):
    # Load detections
    det_dir = osp.join(args.data_root, args.det_dir, "det2")
    detections = load_mot_detections(det_dir)
    total_frames = len(detections)
    logger.info(f"Loaded {total_frames} frames of detections from {det_dir}")

    # Initialize ReID extractor
    extractor = FeatureExtractor(
        model_name="osnet_x1_0",
        model_path=args.reid_model,
        device="cuda"
    ) if args.with_reid else None

    # Initialize tracker
    tracker = Deep_EIoU(args, frame_rate=30)
    results = []

    # Process each frame with tqdm progress bar
    img_dir = osp.join(args.data_root, args.img_dir)
    for frame_id in tqdm(range(1, total_frames + 1), total=total_frames, desc="Tracking progress"):
        frame_path = osp.join(img_dir, f"{frame_id:06d}.jpg")
        if not osp.exists(frame_path):
            logger.warning(f"Frame {frame_id} image missing: {frame_path}")
            continue

        frame = cv2.imread(frame_path)
        current_dets = detections.get(frame_id, np.empty((0, 5)))

        if len(current_dets) > 0:
            # Extract ReID features
            if args.with_reid:
                cropped_imgs = [
                    frame[int(y1):int(y2), int(x1):int(x2)]
                    for x1, y1, x2, y2, _ in current_dets
                ]
                embs = extractor(cropped_imgs).cpu().numpy()
            else:
                embs = None

            # Update tracker
            online_targets = tracker.update(current_dets, embs)

            # Save results (MOT format)
            for t in online_targets:
                tlwh = t.last_tlwh
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    results.append(
                        f"{frame_id},{t.track_id},{tlwh[0]:.1f},{tlwh[1]:.1f},{tlwh[2]:.1f},{tlwh[3]:.1f},{t.score:.2f},-1,-1,-1\n"
                    )

    # Save tracking results
    output_path = osp.join(args.data_root, args.output)
    with open(output_path, "w") as f:
        f.writelines(results)
    logger.success(f"Saved tracking results to {output_path}")

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
