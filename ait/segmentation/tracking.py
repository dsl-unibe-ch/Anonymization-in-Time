"""
Temporal tracking (IoU-based mask matching) and gap propagation.
"""

import numpy as np
import cv2

from .mask_ops import mask_entry_to_crop, make_mask_entry


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


def match_masks_across_frames(results_list, iou_threshold=0.3):
    """
    Match masks across frames by IoU.

    Returns list of tracks where each track is a list of (list_idx, mask_idx) pairs.
    """
    if not results_list or len(results_list) == 0:
        return []

    tracks = []

    first_results = results_list[0][1]
    if first_results and 'boxes' in first_results:
        for mask_idx in range(len(first_results['boxes'])):
            tracks.append([(0, mask_idx)])

    for frame_idx in range(1, len(results_list)):
        curr_results = results_list[frame_idx][1]

        if not curr_results or 'boxes' not in curr_results:
            continue

        curr_boxes = curr_results['boxes']
        matched_tracks = set()
        matched_masks = set()

        for mask_idx, curr_box in enumerate(curr_boxes):
            best_track_idx = -1
            best_iou = iou_threshold

            for track_idx, track in enumerate(tracks):
                if track_idx in matched_tracks:
                    continue

                last_frame, last_mask_idx = track[-1]
                last_results = results_list[last_frame][1]
                last_box = last_results['boxes'][last_mask_idx]

                # Reject if box areas are too different (e.g. small group-chat
                # icon overlapping a large chat-list icon during transition)
                curr_area = (curr_box[2] - curr_box[0]) * (curr_box[3] - curr_box[1])
                last_area = (last_box[2] - last_box[0]) * (last_box[3] - last_box[1])
                if min(curr_area, last_area) < 0.75 * max(curr_area, last_area):
                    continue

                iou = calculate_iou(curr_box, last_box)

                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = track_idx

            if best_track_idx >= 0:
                tracks[best_track_idx].append((frame_idx, mask_idx))
                matched_tracks.add(best_track_idx)
                matched_masks.add(mask_idx)
            else:
                tracks.append([(frame_idx, mask_idx)])
                matched_masks.add(mask_idx)

    return tracks


def _boxes_overlap(box_a, box_b):
    """Return True if two XYXY boxes overlap with positive area."""
    return (box_a[0] < box_b[2] and box_b[0] < box_a[2] and
            box_a[1] < box_b[3] and box_b[1] < box_a[3])


def propagate_missing_masks(results_list, tracks, max_gap=5):
    """
    Fill gaps in tracks by interpolating masks.

    Before placing a propagated mask, checks that it doesn't overlap any
    real (original) detection in that frame. If it does, the mask likely
    moved and the propagated position is wrong — skip it.
    """
    # Snapshot original boxes per frame before we start adding propagated ones
    original_boxes = {}
    for frame_idx, (_, frame_results, _) in enumerate(results_list):
        if frame_results and 'boxes' in frame_results:
            original_boxes[frame_idx] = list(frame_results['boxes'])
        else:
            original_boxes[frame_idx] = []

    filled_count = 0
    skipped_count = 0

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

                # Skip if propagated box overlaps any real detection in this frame
                overlaps_real = False
                for real_box in original_boxes.get(missing_frame, []):
                    if _boxes_overlap(interp_box, real_box):
                        overlaps_real = True
                        break
                if overlaps_real:
                    skipped_count += 1
                    continue

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

    if skipped_count > 0:
        print(f"  Skipped {skipped_count} propagated mask(s) that overlapped real detections")

    return filled_count
