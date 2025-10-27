import argparse
import json
import os
import itertools
import glob
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm
import torch
from torchvision.io import write_video as _wv
import random

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any, Optional

COLORS = [(255, 64, 64), (0, 0, 255), (127, 255, 0), (255, 97, 3), (220, 20, 60),
          (255, 185, 15), (255, 20, 147), (255, 105, 180), (60, 179, 113)]

def write_video(path, frames, fps, order):
    if order.lower() == 'nchw':
        frames = frames.permute(0, 2, 3, 1)
    elif order.lower() == 'nhwc':
        frames = frames
    else:
        raise Exception("Illegal order, one of nchw or nhwc")
    
    _wv(
        path,
        frames,
        fps,
        video_codec="libx264",
        options={"crf": "18"}
    )


from typing import Dict, List, Tuple, Any, Optional

Point = Tuple[float, float]
BBox  = Tuple[float, float, float, float]

def emit_trajectories_by_id(
    frames: List[Dict[str, Any]],
    key: str = "gt_annotation",
    fill_missing: Optional[Point] = None,
) -> Dict[str, List[Optional[Point]]]:
    """
    Returns {object_id: [(cx, cy) or None]*len(frames)} keyed by `gt_annotation`.
    """
    T = len(frames)

    # discover all unique ids that ever appear
    ids = []
    seen = set()
    for f in frames:
        for lab in f.get("labels", []):
            oid = str(lab.get(key, ""))
            if oid and oid not in seen:
                seen.add(oid)
                ids.append(oid)

    traj: Dict[str, List[Optional[Point]]] = {oid: [fill_missing]*T for oid in ids}

    for t, f in enumerate(frames):
        for lab in f.get("labels", []):
            oid = str(lab.get(key, ""))
            if not oid:
                continue
            b = lab.get("box2d", {})
            x1, y1, x2, y2 = b.get("x1"), b.get("y1"), b.get("x2"), b.get("y2")
            if None in (x1, y1, x2, y2):
                continue
            cx = (float(x1) + float(x2)) * 0.5
            cy = (float(y1) + float(y2)) * 0.5
            traj[oid][t] = (cx, cy)

    return traj


def emit_bboxes_by_id(
    frames: List[Dict[str, Any]],
    key: str = "gt_annotation",
    fill_missing: Optional[BBox] = None,
) -> Dict[str, List[Optional[BBox]]]:
    """
    Returns {object_id: [(x1, y1, x2, y2) or None]*len(frames)} keyed by `gt_annotation`.
    """
    T = len(frames)

    # discover all unique ids that ever appear
    ids = []
    seen = set()
    for f in frames:
        for lab in f.get("labels", []):
            oid = str(lab.get(key, ""))
            if oid and oid not in seen:
                seen.add(oid)
                ids.append(oid)

    seq: Dict[str, List[Optional[BBox]]] = {oid: [fill_missing]*T for oid in ids}

    for t, f in enumerate(frames):
        for lab in f.get("labels", []):
            oid = str(lab.get(key, ""))
            if not oid:
                continue
            b = lab.get("box2d", {})
            if not all(k in b for k in ("x1", "y1", "x2", "y2")):
                continue
            bb = (float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]))
            seq[oid][t] = bb

    return seq


def annotate_video(elem):
    video_path, annotation, output_video = elem
    
    video = VideoDecoder(video_path, dimension_order='NHWC')
    metadata = video.metadata
    frames = video.get_frames_in_range(start=0, stop=metadata.num_frames, step=1).data
    
    frames = frames.numpy().astype(np.uint8)
    
    bboxes = emit_bboxes_by_id(annotation, key='gt_annotation')
    trajectories = emit_trajectories_by_id(annotation, key='gt_annotation')

    colors = random.sample(COLORS, len(bboxes))
    colors = dict(zip(bboxes.keys(), colors))

    for idx, _ in enumerate(annotation):
        if idx > len(frames) - 1:
            continue

        frame = frames[idx]
        for key in bboxes:
            if key == 'object hand':
                continue
            
            if bboxes[key][idx] is None:
                continue
            
            x1, y1, x2, y2  = bboxes[key][idx]
            
            cv2.rectangle(frame, tuple([int(x1), int(y1)]), tuple([int(x2), int(y2)]), colors[key], 2)
            
            trajectory = trajectories[key][:idx]
            prevx, prevy = None, None
            for point in trajectory:
                if point:
                    x, y = point
                    x, y = int(x), int(y)
                    cv2.circle(frame, center=(x, y), radius=2, color=colors[key], thickness=-1)
                    if prevx:
                        cv2.line(frame, pt1=(x, y), pt2=(prevx, prevy), color=colors[key], thickness=1)

                    prevx, prevy = x,y

    _, H, W, _ = frames.shape
    frames = frames[:, :(H // 2) * 2,:(W // 2) * 2,:]
    
    frames = torch.from_numpy(frames)
    write_video(str(output_video), frames, 12, 'nhwc')


def annotate_videos(videos_dir, annotation_path, output_dir):
    annotations = json.load(open(annotation_path))
    
    n_elems = 10000
    keys = sorted(annotations.keys())[:n_elems]
    
    videos_dir = Path(videos_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)
    
    assert videos_dir.exists()

    elems = []
    for video_name in tqdm(keys):
        annotation = sorted(annotations[video_name], key=lambda x:x['name'])
        video_path = videos_dir / f'{video_name}.webm'
        output_video = output_dir / f"{video_name}.mp4"
        if not output_video.exists():
            elems.append([video_path, annotation, output_video])

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(tqdm(ex.map(annotate_video, elems), total=len(elems)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--annotations', required=True)
    
    args = parser.parse_args()
    annotate_videos(args.videos, args.annotations, args.out)