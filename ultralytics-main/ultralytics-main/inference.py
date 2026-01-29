import cv2
import os
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import torch
from torchvision.ops import nms  


video_path = "/ssd1/ron_soccer/ultralytics-main/dataset/Challenge Videos/132831.mp4"
output_video_path = '/ssd2/ron_soccer/ultralytics-main/132831.mp4'
output_txt_path = '/ssd2/ron_soccer/output1/132831.txt'
os.makedirs('output', exist_ok=True)


model = YOLO("/ssd1/ron_soccer/ultralytics-main/finetune_best.pt")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"無法開啟影片：{video_path}")


total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


f = open(output_txt_path, 'w')
frame_id = 1


with tqdm(total=total_frames, desc="TTA + Multi-Scale + NMS 推論中", unit="frame") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes_all, scores_all, labels_all = [], [], []

       
        for size in [1280, 1920]:
            results = model.predict(
                source=frame,
                conf=0.6,
                iou=0.5,
                imgsz=size,
                augment=True,
                rect=True,
                half=False,
                verbose=False
            )
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                boxes_all.append([x1, y1, x2, y2])
                scores_all.append(conf)
                labels_all.append(cls)

       
        boxes_tensor = torch.tensor(boxes_all)
        scores_tensor = torch.tensor(scores_all)
        labels_tensor = torch.tensor(labels_all)

        
        final_indices = []
        for cls in torch.unique(labels_tensor):
            cls_mask = labels_tensor == cls
            cls_boxes = boxes_tensor[cls_mask]
            cls_scores = scores_tensor[cls_mask]

            keep = nms(cls_boxes, cls_scores, iou_threshold=0.5)
            final_indices.extend(cls_mask.nonzero(as_tuple=True)[0][keep].tolist())

      
        annotated_frame = frame.copy()
        for idx in final_indices:
            box = boxes_tensor[idx]
            score = scores_tensor[idx].item()
            label = labels_tensor[idx].item()
            x1, y1, x2, y2 = map(int, box.tolist())
            w, h = x2 - x1, y2 - y1

            f.write(f"{frame_id} -1 {x1:.2f} {y1:.2f} {w:.2f} {h:.2f} {score:.2f} {label} -1 -1\n")

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'{score:.2f}', (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(annotated_frame)
        frame_id += 1
        pbar.update(1)


f.close()
cap.release()
out.release()
cv2.destroyAllWindows()