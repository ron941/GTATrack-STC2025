# This script produces tracklets given tracking results and original sequence frame as RGB images.
import argparse
from torchreid.utils import FeatureExtractor

import os
from tqdm import tqdm
from loguru import logger
from PIL import Image

import pickle
import numpy as np
import glob

import torch
import torchvision.transforms as T
from Tracklet import Tracklet

def main(model_path, data_path, pred_dir, tracker):
    # load feature extractor:
    val_transforms = T.Compose([
            T.Resize([256, 128]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    split = os.path.basename(data_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path = model_path,
        device=device
    )

    output_dir = os.path.join(os.path.dirname(pred_dir), f'{tracker}_Tracklets_{split}')  # output directory for sequences' tracklets
    os.makedirs(output_dir, exist_ok=True)

    seqs = sorted([file for file in os.listdir(pred_dir) if file.endswith('.txt')])

    for s_id, seq in tqdm(enumerate(seqs, 1), total=len(seqs), desc='Processing Seqs'):
        seq = seq.replace('.txt','')
        imgs = sorted(glob.glob(os.path.join(data_path, seq, 'img1', '*')))   # assuming data is organized in MOT convention
        track_res = np.genfromtxt(os.path.join(pred_dir, f'{seq}.txt'),dtype=float, delimiter=',')

        last_frame = int(track_res[-1][0])
        seq_tracks = {}

        for frame_id in range(1, last_frame+1):
            if frame_id%100 == 0:
                logger.info(f'Processing frame {frame_id}/{last_frame}')

            # query all track_res for current frame
            inds = track_res[:,0] == frame_id
            frame_res = track_res[inds]
            img = Image.open(imgs[int(frame_id)-1])
            
            input_batch = None    # input batch to speed up processing
            tid2idx = {}
            
            # NOTE MOT annotation format:
            # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
            for idx, (frame, track_id, l, t, w, h, score, _, _, _) in enumerate(frame_res):
                # Update tracklet with detection
                bbox = [l, t, w, h]
                if track_id not in seq_tracks:
                    seq_tracks[track_id] = Tracklet(track_id, frame, score, bbox)
                else:
                    seq_tracks[track_id].append_det(frame, score, bbox)
                tid2idx[track_id] = idx

                im = img.crop((l, t, l+w, t+h)).convert('RGB')
                im = val_transforms(im).unsqueeze(0)
                if input_batch is None:
                        input_batch = im
                else:
                    input_batch = torch.cat([input_batch, im], dim=0)
            
            if input_batch is not None:
                features = extractor(input_batch)    # len(features) == len(frame_res)
                feats = features.cpu().detach().numpy()
                
                # update tracklets with feature
                for tid, idx in tid2idx.items():
                    feat = feats[tid2idx[tid]]
                    feat /= np.linalg.norm(feat)
                    seq_tracks[tid].append_feat(feat)
            else:
                print(f"No detection at frame: {frame_id}")
        
        # save seq_tracks into pickle file
        track_output_path = os.path.join(output_dir,  f'{seq}.pkl')
        with open(track_output_path, 'wb') as f:
            pickle.dump(seq_tracks, f)
        logger.info(f"save tracklets info to {track_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tracklets from tracking results.")
    parser.add_argument('--model_path',
                        type=str,
                        default=os.path.join('..', 'reid_checkpoints', 'sports_model.pth.tar-60'),
                        help="Path to the model file.")
    parser.add_argument('--data_path',
                        type=str,
                        default=r"C:\Users\Ciel Sun\OneDrive - UW\EE 599\SoccerNet\tracking-2023\test",
                        # required=True,
                        help="Directory containing data files.")
    parser.add_argument('--pred_dir',
                        type=str,
                        default=r"C:\Users\Ciel Sun\OneDrive - UW\EE 599\SoccerNet\DeepEIoU_Results\DeepEIoU_Baseline",
                        # required=True,
                        help="Directory containing prediction files.")
    parser.add_argument('--tracker',
                        type=str,
                        default='DeepEIoU',
                        # required=True,
                        help="Name of the tracker.")
    
    args = parser.parse_args()
    
    main(args.model_path, args.data_path, args.pred_dir, args.tracker)