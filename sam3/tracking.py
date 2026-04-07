"""
Temporal tracking (IoU + position-based mask matching) and gap propagation.
"""

import numpy as np
import cv2

from .mask_ops import mask_entry_to_crop, make_mask_entry


def _box_center(box):
    """Return (cx, cy) of an XYXY box."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _box_size(box):
    """Return (w, h) of an XYXY box."""
    return (box[2] - box[0], box[3] - box[1])


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in XYXY format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def match_masks_across_frames(results_list, iou_threshold=0.3, max_stale=50):
    """
    Match masks across frames using IoU + position/size fallback.

    A track stays open for matching for up to `max_stale` frames after its last
    detection.  When IoU is zero (no overlap after a gap), we fall back to
    center-distance + size-similarity to reconnect the track.

    Returns list of tracks where each track is a list of (list_idx, mask_idx) pairs.
    """
    if not results_list or len(results_list) == 0:
        return []

    tracks = []           # each track: list of (list_idx, mask_idx)
    track_last_box = []   # last known box per track
    track_last_frame = [] # last frame_idx per track

    first_results = results_list[0][1]
    if first_results and 'boxes' in first_results:
        for mask_idx in range(len(first_results['boxes'])):
            tracks.append([(0, mask_idx)])
            track_last_box.append(first_results['boxes'][mask_idx])
            track_last_frame.append(0)

    for frame_idx in range(1, len(results_list)):
        curr_results = results_list[frame_idx][1]

        if not curr_results or 'boxes' not in curr_results:
            continue

        curr_boxes = curr_results['boxes']
        matched_tracks = set()
        matched_masks = set()

        # Score every (mask, track) pair, pick best assignments greedily
        candidates = []
        for mask_idx, curr_box in enumerate(curr_boxes):
            curr_cx, curr_cy = _box_center(curr_box)
            curr_w, curr_h = _box_size(curr_box)

            for track_idx in range(len(tracks)):
                if track_idx in matched_tracks:
                    continue

                # Skip tracks that have been stale too long
                stale_frames = frame_idx - track_last_frame[track_idx]
                if stale_frames > max_stale:
                    continue

                last_box = track_last_box[track_idx]
                iou = calculate_iou(curr_box, last_box)

                if iou >= iou_threshold:
                    # High confidence match via IoU
                    candidates.append((mask_idx, track_idx, 2.0 + iou))
                else:
                    # Fallback: center distance + size similarity
                    last_cx, last_cy = _box_center(last_box)
                    last_w, last_h = _box_size(last_box)

                    # Normalized distance: relative to box size
                    avg_size = max(1.0, (last_w + last_h + curr_w + curr_h) / 4.0)
                    dist = np.sqrt((curr_cx - last_cx)**2 + (curr_cy - last_cy)**2)
                    norm_dist = dist / avg_size

                    # Size ratio (closer to 1.0 = better)
                    size_ratio = min(curr_w, last_w) / max(curr_w, last_w, 1.0) * \
                                 min(curr_h, last_h) / max(curr_h, last_h, 1.0)

                    # Accept if close enough in position and similar size
                    if norm_dist < 1.5 and size_ratio > 0.5:
                        # Score: higher is better (invert distance, weight size)
                        score = size_ratio / (1.0 + norm_dist)
                        candidates.append((mask_idx, track_idx, score))

        # Greedy assignment: best scores first
        candidates.sort(key=lambda c: -c[2])
        for mask_idx, track_idx, score in candidates:
            if mask_idx in matched_masks or track_idx in matched_tracks:
                continue
            tracks[track_idx].append((frame_idx, mask_idx))
            track_last_box[track_idx] = curr_boxes[mask_idx]
            track_last_frame[track_idx] = frame_idx
            matched_tracks.add(track_idx)
            matched_masks.add(mask_idx)

        # Start new tracks for unmatched masks
        for mask_idx in range(len(curr_boxes)):
            if mask_idx not in matched_masks:
                tracks.append([(frame_idx, mask_idx)])
                track_last_box.append(curr_boxes[mask_idx])
                track_last_frame.append(frame_idx)

    return tracks


def propagate_missing_masks(results_list, tracks, max_gap=5):
    """Fill gaps in tracks by interpolating masks."""
    filled_count = 0

    for track in tracks:
        if len(track) < 2:
            continue

        for i in range(len(track) - 1):
            curr_frame, curr_mask = track[i]
            next_frame, next_mask = track[i + 1]
            gap_size = next_frame - curr_frame - 1

            if gap_size <= 0 or gap_size > max_gap:
                continue

            curr_results = results_list[curr_frame][1]
            next_results = results_list[next_frame][1]

            curr_box = curr_results['boxes'][curr_mask]
            next_box = next_results['boxes'][next_mask]
            curr_mask_data = curr_results['masks'][curr_mask]
            next_mask_data = next_results['masks'][next_mask]

            curr_crop, curr_bbox = mask_entry_to_crop(curr_mask_data)
            next_crop, next_bbox = mask_entry_to_crop(next_mask_data)
            pack_output = isinstance(curr_mask_data, dict) and "packed" in curr_mask_data

            for gap_idx in range(1, gap_size + 1):
                missing_frame = curr_frame + gap_idx
                alpha = gap_idx / (gap_size + 1)

                interp_box = curr_box * (1 - alpha) + next_box * alpha

                x1, y1, x2, y2 = interp_box
                target_w = max(1, int(round(x2 - x1 + 1)))
                target_h = max(1, int(round(y2 - y1 + 1)))
                curr_resized = cv2.resize(curr_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                next_resized = cv2.resize(next_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                interp_mask = curr_resized * (1 - alpha) + next_resized * alpha
                interp_bool = interp_mask > 0.5
                bbox_int = [int(round(v)) for v in interp_box]
                mask_entry = make_mask_entry(interp_bool, bbox_int, pack_bits=pack_output)

                missing_results = results_list[missing_frame][1]
                if not missing_results or 'boxes' not in missing_results:
                    missing_results = {'boxes': [], 'masks': [], 'scores': []}
                    results_list[missing_frame] = (
                        results_list[missing_frame][0],
                        missing_results,
                        results_list[missing_frame][2]
                    )

                if not isinstance(missing_results['boxes'], list):
                    missing_results['boxes'] = list(missing_results['boxes'])
                    missing_results['masks'] = list(missing_results['masks'])
                    missing_results['scores'] = list(missing_results['scores'])

                missing_results['boxes'].append(np.array(interp_box, dtype=np.float32))
                missing_results['masks'].append(mask_entry)
                missing_results['scores'].append(0.5)

                filled_count += 1

    return filled_count
