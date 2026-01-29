import cv2
import os
from tqdm import tqdm

video_path = "/ssd1/ron_soccer/ultralytics-main/dataset/Challenge Videos/128057.mp4"
output_dir = "/ssd1/ron_soccer/Deep-EIoU/Deep-EIoU/data/football/img1"  # 切出對齊的 img1 資料夾

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with tqdm(total=total_frames, desc="Extracting frames") as pbar:
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        save_path = os.path.join(output_dir, f"{frame_idx+1:06d}.jpg")
        cv2.imwrite(save_path, frame)

        frame_idx += 1
        pbar.update(1)

cap.release()
print(f" Done: Extracted {frame_idx} frames to {output_dir}")
