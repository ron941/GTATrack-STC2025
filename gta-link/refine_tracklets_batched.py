import numpy as np
import os
import torch
import pickle
import random

from collections import defaultdict
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
import concurrent.futures

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from Tracklet import Tracklet

import argparse

def detect_id_switch(embs, eps=None, min_samples=None, max_clusters=None):
    """
    Detects identity switches within a tracklet using clustering.

    Args:
        embs (list of numpy arrays): A list where each element is a numpy array representing an embedding.
                                     Each embedding has the same dimensionality.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        bool: True if an identity switch is detected, otherwise False.
    """
    if len(embs) > 15000:
        embs = embs[1::2]

    embs = np.stack(embs)
    
    # Standardize the embeddings
    scaler = StandardScaler()
    embs_scaled = scaler.fit_transform(embs)

    # Apply DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embs_scaled)
    labels = db.labels_

    # Count the number of clusters (excluding noise)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    if -1 in labels and len(unique_labels) > 1:
        # Find the cluster centers
        cluster_centers = np.array([embs_scaled[labels == label].mean(axis=0) for label in unique_labels])
        # if len(unique_labels) == 1 and unique_labels[0] == -1:      # debug line, delete later
        #     print("Cluster centers:\n", cluster_centers)            # debug line, delete later
        #     print("Labels:\n", labels)                              # debug line, delete later
        # Assign noise points to the nearest cluster
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            distances = cdist([embs_scaled[idx]], cluster_centers, metric='cosine')
            nearest_cluster = np.argmin(distances)
            labels[idx] = list(unique_labels)[nearest_cluster]
    
    n_clusters = len(unique_labels)

    if max_clusters and n_clusters > max_clusters:
        # Merge clusters to ensure the number of clusters does not exceed max_clusters
        while n_clusters > max_clusters:
            cluster_centers = np.array([embs_scaled[labels == label].mean(axis=0) for label in unique_labels])
            distance_matrix = cdist(cluster_centers, cluster_centers, metric='cosine')
            np.fill_diagonal(distance_matrix, np.inf)  # Ignore self-distances
            
            # Find the closest pair of clusters
            min_dist_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            cluster_to_merge_1, cluster_to_merge_2 = unique_labels[min_dist_idx[0]], unique_labels[min_dist_idx[1]]

            # Merge the clusters
            labels[labels == cluster_to_merge_2] = cluster_to_merge_1
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]
            n_clusters = len(unique_labels)

    return n_clusters > 1, labels

def split_tracklets(tmp_trklets, eps=None, max_k=None, min_samples=None, len_thres=None):
    """
    Splits each tracklet into multiple tracklets based on an internal distance threshold.

    Args:
        tmp_trklets (dict): Dictionary of tracklets to be processed.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        inner_dist_thres (float): Threshold for the average inner distance to decide splitting.
        len_thres (int): Length threshold to filter out short tracklets.
        max_k (int): Maximum number of clusters to consider.

    Returns:
        dict: New dictionary of tracklets after splitting.
    """
    new_id = max(tmp_trklets.keys()) + 1
    tracklets = defaultdict()
    # Splitting algorithm to process every tracklet in a sequence
    for tid in tqdm(sorted(list(tmp_trklets.keys())), total=len(tmp_trklets), desc="Splitting tracklets"):
        # print("Track ID:\n", tid)               # debug line, delete later
        trklet = tmp_trklets[tid]
        if len(trklet.times) < len_thres:  # NOTE: Set tracklet length threshold to filter out short ones
            tracklets[tid] = trklet
        else:
            embs = np.stack(trklet.features)
            frames = np.array(trklet.times)
            bboxes = np.stack(trklet.bboxes)
            scores = np.array(trklet.scores)
            # Perform DBSCAN clustering
            id_switch_detected, clusters = detect_id_switch(embs, eps=eps, min_samples=min_samples, max_clusters=max_k)
            if not id_switch_detected:
                tracklets[tid] = trklet
            else:
                unique_labels = set(clusters)

                for label in unique_labels:
                    if label == -1:
                        continue  # Skip noise points
                    tmp_embs = embs[clusters == label]
                    tmp_frames = frames[clusters == label]
                    tmp_bboxes = bboxes[clusters == label]
                    tmp_scores = scores[clusters == label]
                    assert new_id not in tmp_trklets
                    # TODO: Create new tracklet object
                    tracklets[new_id] = Tracklet(new_id, tmp_frames.tolist(), tmp_scores.tolist(), tmp_bboxes.tolist(), feats=tmp_embs.tolist())
                    new_id += 1

    assert len(tracklets) >= len(tmp_trklets)
    return tracklets

def find_consecutive_segments(track_times):
    """
    Identifies and returns the start and end indices of consecutive segments in a list of times.

    Args:
        track_times (list): A list of frame times (integers) representing when a tracklet was detected.

    Returns:
        list of tuples: Each tuple contains two integers (start_index, end_index) representing the start and end of a consecutive segment.
    """
    segments = []
    start_index = 0
    end_index = 0
    for i in range(1, len(track_times)):
        if track_times[i] == track_times[end_index] + 1:
            end_index = i
        else:
            segments.append((start_index, end_index))
            start_index = i
            end_index = i
    segments.append((start_index, end_index))
    return segments

def query_subtracks(seg1, seg2, track1, track2):
    """
    Processes and pairs up segments from two different tracks to form valid subtracks based on their temporal alignment.

    Args:
        seg1 (list of tuples): List of segments from the first track where each segment is a tuple of start and end indices.
        seg2 (list of tuples): List of segments from the second track similar to seg1.
        track1 (Tracklet): First track object containing times and bounding boxes.
        track2 (Tracklet): Second track object similar to track1.

    Returns:
        list: Returns a list of subtracks which are either segments of track1 or track2 sorted by time.
    """
    subtracks = []  # List to store valid subtracks
    while seg1 and seg2:  # Continue until seg1 or seg1 is empty
        s1_start, s1_end = seg1[0]  # Get the start and end indices of the first segment in seg1
        s2_start, s2_end = seg2[0]  # Get the start and end indices of the first segment in seg2
        '''Optionally eliminate false positive subtracks
        if (s1_end - s1_start + 1) < 30:
            seg1.pop(0)  # Remove the first element from seg1
            continue
        if (s2_end - s2_start + 1) < 30:
            seg2.pop(0)  # Remove the first element from seg2
            continue
        '''

        # subtrack_1 = get_subtrack(track1, s1_start, s1_end)  # Extract subtrack from track 1
        # subtrack_2 = get_subtrack(track2, s2_start, s2_end)  # Extract subtrack from track 2
        subtrack_1 = track1.extract(s1_start, s1_end)
        subtrack_2 = track2.extract(s2_start, s2_end)

        s1_startFrame = track1.times[s1_start]  # Get the starting frame of subtrack 1
        s2_startFrame = track2.times[s2_start]  # Get the starting frame of subtrack 2

        # print("track 1 and 2 start frame:", s1_startFrame, s2_startFrame)
        # print("track 1 and 2 end frame:", track1.times[s1_end], track2.times[s2_end])

        if s1_startFrame < s2_startFrame:  # Compare the starting frames of the two subtracks
            assert track1.times[s1_end] <= s2_startFrame
            subtracks.append(subtrack_1)
            subtracks.append(subtrack_2)
        else:
            assert s1_startFrame >= track2.times[s2_end]
            subtracks.append(subtrack_2)
            subtracks.append(subtrack_1)
        seg1.pop(0)
        seg2.pop(0)
    
    seg_remain = seg1 if seg1 else seg2
    track_remain = track1 if seg1 else track2
    while seg_remain:
        s_start, s_end = seg_remain[0]
        if(s_end - s_start) < 30:
            seg_remain.pop(0)
            continue
        # subtracks.append(get_subtrack(track_remain, s_start, s_end))
        subtracks.append(track_remain.extract(s_start, s_end))
        seg_remain.pop(0)
    
    return subtracks  # Return the list of valid subtracks sorted ascending temporally

def get_subtrack(track, s_start, s_end):
    """
    Extracts a subtrack from a given track.

    Args:
    track (STrack): The original track object from which the subtrack is to be extracted.
    s_start (int): The starting index of the subtrack.
    s_end (int): The ending index of the subtrack.

    Returns:
    STrack: A subtrack object extracted from the original track object, containing the specified time intervals
            and bounding boxes. The parent track ID is also assigned to the subtrack.
    """
    subtrack = Tracklet()
    subtrack.times = track.times[s_start : s_end + 1]
    subtrack.bboxes = track.bboxes[s_start : s_end + 1]
    subtrack.parent_id = track.track_id

    return subtrack

def get_spatial_constraints(tid2track, factor):
    """
    Calculates and returns the maximal spatial constraints for bounding boxes across all tracks.

    Args:
        tid2track (dict): Dictionary mapping track IDs to their respective track objects.
        factor (float): Factor by which to scale the calculated x and y ranges.

    Returns:
        tuple: Maximal x and y range scaled by the given factor.
    """

    min_x = float('inf')
    max_x = -float('inf')
    min_y = float('inf')
    max_y = -float('inf')

    for track in tid2track.values():
        for bbox in track.bboxes:
            assert len(bbox) == 4
            x, y, w, h = bbox[0:4]  # x, y is coordinate of top-left point of bounding box
            x += w / 2  # get center point
            y += h / 2  # get center point
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    x_range = abs(max_x - min_x) * factor
    y_range = abs(max_y - min_y) * factor

    return x_range, y_range

def display_Dist(seq2Dist, seq_name = None, isMerged=False, isSplit=False):
    """
    Displays a heatmap for the distances between tracklets for one or more sequences.

    Args:
        seq2Dist (dict): A dictionary mapping sequence names to their corresponding distance matrices.
        seq_name (str, optional): Specific sequence name to display the heatmap for. If None, displays for all sequences.
        isMerged (bool): Flag indicating whether the distances are post-merge.
        isSplit (bool): Flag indicating whether the distances are post-split.

    """
    split_info = " After Split" if isSplit else " Before Split"
    merge_info = " After Merge" if isMerged else " Before Merge"
    info = split_info + merge_info
    if seq_name is None:          # Display all sequences' distance maps if no sequence is specified
        seqs = list(seq2Dist.keys())
        for i in range(len(seqs)):
            seq = seqs[i]
            Dist = seq2Dist[seq]
            # fig, ax = plt.subplots()
            # ticks = np.arange(len(Dist))
            # im, cbar = heatmap(Dist, ticks, ticks, ax=ax, cmap='binary', cbarlabel='distance')
            # texts = annotate_heatmap(im, valfmt="{x:.2f}")
            # fig.tight_layout()
            # plt.show()

            plt.imshow(Dist, cmap='binary')
            plt.colorbar()
            plt.title(seq + info)
            plt.show()
    else:
        assert seq_name in set(seq2Dist.keys())
        Dist = seq2Dist[seq_name]
        plt.imshow(Dist, cmap='binary')
        plt.colorbar()
        plt.title(seq_name + info)
        plt.show()

def calculate_distance(i, j, track1_id, track2_id, track1, track2, Dist):
    if j < i:
        return (i, j, Dist[j][i])
    else:
        return (i, j, get_distance(track1_id, track2_id, track1, track2))

# Parallel get_distance_matrix function
def get_distance_matrix_concurrent(tid2track):
    num_tracks = len(tid2track)
    Dist = np.zeros((num_tracks, num_tracks))

    

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, (track1_id, track1) in enumerate(tid2track.items()):
            for j, (track2_id, track2) in enumerate(tid2track.items()):
                futures.append(executor.submit(calculate_distance, i, j, track1_id, track2_id, track1, track2, Dist))

        for future in concurrent.futures.as_completed(futures):
            i, j, distance = future.result()
            Dist[i][j] = distance

    return Dist

def get_distance_matrix(tid2track):
    """
    Constructs and returns a distance matrix between all tracklets based on overlapping times and feature similarities.

    Args:
        tid2track (dict): Dictionary mapping track IDs to their respective track objects.

    Returns:
        ndarray: A square matrix where each element (i, j) represents the calculated distance between track i and track j.
    """
    # print("number of tracks:", len(tid2track))
    Dist = np.zeros((len(tid2track), len(tid2track)))

    for i, (track1_id, track1) in enumerate(tid2track.items()):
        assert len(track1.times) == len(track1.bboxes)
        for j, (track2_id, track2) in enumerate(tid2track.items()):
            if j < i:
                Dist[i][j] = Dist[j][i]
            else:
                Dist[i][j] = get_distance(track1_id, track2_id, track1, track2)
    return Dist

def get_distance(track1_id, track2_id, track1, track2):
    """
    Calculates the cosine distance between two tracks using PyTorch for efficient computation.

    Args:
        track1_id (int): ID of the first track.
        track2_id (int): ID of the second track.
        track1 (Tracklet): First track object.
        track2 (Tracklet): Second track object.

    Returns:
        float: Cosine distance between the two tracks.
    """
    assert track1_id == track1.track_id and track2_id == track2.track_id   # debug line
    # doesOverlap = (track1_id != track2_id)
    # if doesOverlap:
    #     track1_times = set(track1.times)
    #     track2_times = set(track2.times)
    #     doesOverlap = len(track1_times.intersection(track2_times)) > 0
    doesOverlap = False
    if (track1_id != track2_id):
        doesOverlap = set(track1.times) & set(track2.times)
    if doesOverlap:
        return 1                # make the cosine distance between two tracks maximum, max = 1
    else:
        # calculate cosine distance between two tracks based on features
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        track1_features_tensor = torch.tensor(np.stack(track1.features), dtype=torch.float32).to(device)
        track2_features_tensor = torch.tensor(np.stack(track2.features), dtype=torch.float32).to(device)
        count1 = len(track1_features_tensor)
        count2 = len(track2_features_tensor)

        cos_sim_Numerator = torch.matmul(track1_features_tensor, track2_features_tensor.T)
        track1_features_dist = torch.norm(track1_features_tensor, p=2, dim=1, keepdim=True)
        track2_features_dist = torch.norm(track2_features_tensor, p=2, dim=1, keepdim=True)
        cos_sim_Denominator = torch.matmul(track1_features_dist, track2_features_dist.T)
        cos_Dist = 1 - cos_sim_Numerator / cos_sim_Denominator
        
        total_cos_Dist = cos_Dist.sum()
        result = total_cos_Dist / (count1 * count2)
        return result

def check_spatial_constraints(trk_1, trk_2, max_x_range, max_y_range):
    """
    Checks if two tracklets meet spatial constraints for potential merging.

    Args:
        trk_1 (Tracklet): The first tracklet object containing times and bounding boxes.
        trk_2 (Tracklet): The second tracklet object containing times and bounding boxes, to be evaluated
                        against trk_1 for merging possibility.
        max_x_range (float): The maximum allowed distance in the x-coordinate between the end of trk_1 and
                             the start of trk_2 for them to be considered for merging.
        max_y_range (float): The maximum allowed distance in the y-coordinate under the same conditions as
                             the x-coordinate.

    Returns:
        bool: True if the spatial constraints are met (the tracklets are close enough to consider merging),
              False otherwise.
    """
    inSpatialRange = True
    seg_1 = find_consecutive_segments(trk_1.times)
    seg_2 = find_consecutive_segments(trk_2.times)
    '''Debug
    assert((len(seg_1) + len(seg_2)) > 1)         # debug line, delete later
    print(seg_1)                                  # debug line, delete later
    print(seg_2)                                  # debug line, delete later
    '''
    
    subtracks = query_subtracks(seg_1, seg_2, trk_1, trk_2)
    # assert(len(subtracks) > 1)                    # debug line, delete later
    subtrack_1st = subtracks.pop(0)
    while subtracks:
        subtrack_2nd = subtracks.pop(0)
        if subtrack_1st.parent_id == subtrack_2nd.parent_id:
            subtrack_1st = subtrack_2nd
            continue
        x_1, y_1, w_1, h_1 = subtrack_1st.bboxes[-1][0 : 4]
        x_2, y_2, w_2, h_2 = subtrack_2nd.bboxes[0][0 : 4]
        x_1 += w_1 / 2
        y_1 += h_1 / 2
        x_2 += w_2 / 2
        y_2 += h_2 / 2
        dx = abs(x_1 - x_2)
        dy = abs(y_1 - y_2)
        # check the distance between exit location of track_1 and enter location of track_2
        if dx > max_x_range or dy > max_y_range:
            inSpatialRange = False
            # print(f"dx={dx}, dy={dy} out of range max_x_range = {max_x_range}, max_y_range  = {max_y_range}")    # debug line, delete later
            break
        else:
            subtrack_1st = subtrack_2nd
    return inSpatialRange

def merge_tracklets_batched(tracklets, seq2Dist, batch_size=50, seq_name=None, max_x_range=None, max_y_range=None, merge_dist_thres=None):
    """
    Merges tracklets in batches based on a distance threshold.
    
    Parameters:
    tracklets (dict): A dictionary of tracklets where keys are tracklet IDs and values are tracklet objects.
    seq2Dist (dict): A dictionary to store distance matrices for sequences.
    batch_size (int): The size of the batches to process at a time.
    seq_name (str): The name of the sequence being processed.
    max_x_range (float): Maximum allowed distance in the x direction for merging.
    max_y_range (float): Maximum allowed distance in the y direction for merging.
    merge_dist_thres (float): Distance threshold below which tracklets should be merged.
    
    Returns:
    dict: The merged tracklets.
    """
    # seq2Dist[seq_name] = Dist                               # save all seqs distance matrix, debug line, delete later

    temp_tracklets = {}
    tracklet_items = list(tracklets.items())

    # Shuffle tracklet_items with a fixed random seed
    # random.seed(42)
    # random.shuffle(tracklet_items)

    # Batched clustering
    # For batch the tracklets into groups of batch_size (last batch could be less than specified batch size):
    #   get batch_Dist with batch_Dist = get_distance_matrix(batched tracklets)
    #   while (np.any(batch_Dist[non_diagonal_mask] < merge_dist_thres)): keep merging
    #   save merged batched tracklets to temp_tracklets
    # After processing all batches, merge all tracklets in temp_tracklets with while (np.any(Dist[non_diagonal_mask] < merge_dist_thres))
    print(f"Batch size: {batch_size}")
    for i in range(0, len(tracklet_items), batch_size):
        batch_tracklets = dict(tracklet_items[i:i+batch_size])
        print(f"Processing batch from index {i} to {min(i+batch_size - 1, len(tracklet_items) - 1)}")
        
        merged_batch_tracklets= merge_tracklets(batch_tracklets, merge_dist_thres, max_x_range, max_y_range)
        print(f"{len(merged_batch_tracklets)} of {batch_size} tracklets left after merging.")
        temp_tracklets.update(merged_batch_tracklets)
    print(f"Merging {len(temp_tracklets)} tracklets after batched processing.")
    print()
    merged_tracklets = merge_tracklets(temp_tracklets, merge_dist_thres, max_x_range, max_y_range)
    
    return merged_tracklets
    

def merge_tracklets(tracklets, merge_dist_thres, max_x_range, max_y_range):
    Dist = get_distance_matrix(tracklets)
    diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
    non_diagonal_mask = ~diagonal_mask
    idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}

    while np.any(Dist[non_diagonal_mask] < merge_dist_thres):
            min_index = np.argmin(Dist[non_diagonal_mask])
            min_value = np.min(Dist[non_diagonal_mask])

            masked_indices = np.where(non_diagonal_mask)
            track1_idx, track2_idx = masked_indices[0][min_index], masked_indices[1][min_index]

            assert min_value == Dist[track1_idx, track2_idx], "Values should match!"

            track1 = tracklets[idx2tid[track1_idx]]
            track2 = tracklets[idx2tid[track2_idx]]

            if not (set(track1.times) & set(track2.times)):
                inSpatialRange = check_spatial_constraints(track1, track2, max_x_range, max_y_range)
            else:
                inSpatialRange = False

            if inSpatialRange:
                track1.features += track2.features
                track1.times += track2.times
                track1.bboxes += track2.bboxes

                tracklets[idx2tid[track1_idx]] = track1
                tracklets.pop(idx2tid[track2_idx])

                Dist = get_distance_matrix(tracklets)
                idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}
                diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
                non_diagonal_mask = ~diagonal_mask
    return tracklets

def save_results(sct_output_path, tracklets):
    """
    Saves the final tracklet results into a specified path.

    Args:
        sct_output_path (str): Path where the results will be saved.
        tracklets (dict): Dictionary of tracklets containing their final states.

    """
    results = []
    for track_id, track in tracklets.items(): # add each track to results
        tid = track.track_id     # Note: it's the same as track_id
        for instance_idx, frame_id in enumerate(track.times):
            bbox = track.bboxes[instance_idx]
            
            results.append(
                [frame_id, tid, bbox[0], bbox[1], bbox[2], bbox[3], 1, -1, -1, -1]
            )
    results = sorted(results, key=lambda x: x[0])
    txt_results = []
    for line in results:
        txt_results.append(
            f"{line[0]},{line[1]},{line[2]:.2f},{line[3]:.2f},{line[4]:.2f},{line[5]:.2f},{line[6]},{line[7]},{line[8]},{line[9]}\n"
            )
    
    # NOTE: uncomment to save results
    with open(sct_output_path, 'w') as f:
        f.writelines(txt_results)
    logger.info(f"save SCT results to {sct_output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Global tracklet association with splitting and connecting.")
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='Dataset name (e.g., SportsMOT, SoccerNet).')
    
    parser.add_argument('--tracker',
                        type=str,
                        required=True,
                        help='Tracker name.')
    
    parser.add_argument('--track_src',
                        type=str,
                        default=r"C:\Users\Ciel Sun\OneDrive - UW\EE 599\SportsMOT\DeepEIoU_results\DeepEIoU_Tracklets_test",
                        required=True,
                        help='Source directory of tracklet pkl files.'
                        )
    
    parser.add_argument('--use_split',
                        action='store_true',
                        help='If using split component.')
    
    parser.add_argument('--min_len',
                        type=int,
                        default=100,
                        help='Minimum length for a tracklet required for splitting.')
    
    parser.add_argument('--eps',
                        type=float,
                        default=0.7,
                        required=True,
                        help='For DBSCAN clustering, the maximum distance between two samples for one to be considered as in the neighborhood of the other.')
    
    parser.add_argument('--min_samples',
                        type=int,
                        default=10,
                        help='The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.')
    
    parser.add_argument('--max_k',
                        type=int,
                        default=3,
                        help='Maximum number of clusters/subtracklets to be output by splitting component.')
    
    parser.add_argument('--use_connect',
                        action='store_true',
                        help='If using connecting component.')
    
    parser.add_argument('--spatial_factor',
                        type=float,
                        default=1.0,
                        help='Factor to adjust spatial distances.')
    
    parser.add_argument('--merge_dist_thres',
                        type=float,
                        default=0.4,
                        required=True,
                        help='Minimum cosine distance between two tracklets for merging.')
    return parser.parse_args()

def main():
    args = parse_args()
    # Determine the process based on the flags
    if args.use_split and args.use_connect:
        process = "Split+Connect"
    elif args.use_split:
        process = "Split"
    elif args.use_connect:
        process = "Connect"
    else:
        raise ValueError("Both use_split and use_connect are false, must at least use connect.")

    seq_tracks_dir = args.track_src
    data_path = os.path.dirname(seq_tracks_dir)
    seqs_tracks = os.listdir(seq_tracks_dir)
    
    tracker = args.tracker
    dataset = args.dataset

    seqs_tracks.sort()
    seq2Dist = dict()

    process_limit = 10000                           # debug line, delete later
    for seq_idx, seq in enumerate(seqs_tracks):
        if seq_idx >= process_limit:            # debug line, delete later
            break                               # debug line, delete later
        # if seq_idx+1 != 1: continue                                          # debug line, delete later
        # print("Seq name:\n", seq)                                           # debug line, delete later
        seq_name = seq.split('.')[0]
        logger.info(f"Processing seq {seq_idx+1} / {len(seqs_tracks)}")
        with open(os.path.join(seq_tracks_dir, seq), 'rb') as pkl_f:
            tmp_trklets = pickle.load(pkl_f)     # dict(key:track id, value:tracklet)

        max_x_range, max_y_range = get_spatial_constraints(tmp_trklets, args.spatial_factor)
        
        # Dist = getDistanceMap(tmp_trklets)
        # seq2Dist[seq_name] = Dist                                              # save all seqs distance matrix, debug line, delete later
        # displayDist(seq2Dist, seq_name, isMerged=False, isSplit=False)         # used to display Dist, debug line, delete later

        if args.use_split:
            print(f"----------------Number of tracklets before splitting: {len(tmp_trklets)}----------------")
            splitTracklets = split_tracklets(tmp_trklets, eps=args.eps, max_k=args.max_k, min_samples=args.min_samples, len_thres=args.min_len)
        else:
            splitTracklets = tmp_trklets
        
        # Dist = get_distance_matrix(splitTracklets)
        # Dist = get_distance_matrix_concurrent(splitTracklets)

        print(f"----------------Number of tracklets before merging: {len(splitTracklets)}----------------")
        
        mergedTracklets = merge_tracklets_batched(splitTracklets, seq2Dist, batch_size=50, seq_name=seq_name, max_x_range=max_x_range, max_y_range=max_y_range, merge_dist_thres=args.merge_dist_thres)

        print(f"----------------Number of tracklets after merging: {len(mergedTracklets)}----------------")
        
        sct_name = f'{tracker}_{dataset}_{process}_eps{args.eps}_minSamples{args.min_samples}_K{args.max_k}_mergeDist{args.merge_dist_thres}_spatial{args.spatial_factor}'
        os.makedirs(os.path.join(data_path, sct_name), exist_ok=True)
        new_sct_output_path = os.path.join(data_path, sct_name, '{}.txt'.format(seq_name))
        save_results(new_sct_output_path, mergedTracklets)

if __name__ == "__main__":
    main()