import argparse
from pathlib import Path

import os, sys
sys.path.append(os.getcwd() + '/ultralytics')

from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import torch

import pycocotools.mask as maskUtils
from ultralytics import YOLO


def automated_measurements(in_dir: Path, verbose: bool):
    model = YOLO('yolov8m-crowding.yaml').load('checkpoints/yolov8m-crowding.pt')
    preds = model.predict(source=in_dir)

    image_dists = {}
    for pred in preds:
        if verbose:
            pred.show()
        labels = pred.boxes.cls
        scores = pred.boxes.conf
        bboxes = pred.boxes.xywh
        keypoints = pred.keypoints.xy
        sizes = pred.sizes

        anterior_idxs = torch.nonzero(
            (labels < 3) & (bboxes[:, 1] >= 200) & (scores >= 0.5)
        )[:, 0]


        ant_boxes = bboxes[anterior_idxs].cpu().numpy()
        ious = maskUtils.iou(ant_boxes, ant_boxes, [0]*len(ant_boxes))
        remove_idxs = np.nonzero(np.triu(ious, 1) >= 0.8)[1]
        mask = np.ones(anterior_idxs.shape[0], dtype=bool)
        mask[remove_idxs] = False
        anterior_idxs = anterior_idxs[mask]

        assert anterior_idxs.shape[0] == 6

        idxs = anterior_idxs[torch.argsort(bboxes[anterior_idxs, 0])]

        keypoints = keypoints[idxs]
        sizes = sizes[idxs]

        fdis = ['46', '45', '44', '43', '42', '41', '31', '32', '33', '34', '35', '36']
        dists = {}
        for i, (fdi1, fdi2) in enumerate(zip(fdis, fdis[1:]), -3):
            if fdi1[1] in '456' or fdi2[1] in '456':
                dists[f'{fdi1}-{fdi2}'] = 0.0
                continue

            dist1 = torch.linalg.norm(keypoints[i, 0] - keypoints[i, 1])
            dist2 = torch.linalg.norm(keypoints[i + 1, 0] - keypoints[i + 1, 1])
            
            factor = sizes[i, 0] / dist1 if dist1 > dist2 else sizes[i + 1, 0] / dist2 # millimeter per pixel
            
            dist = factor * torch.linalg.norm(keypoints[i, 1 if fdi1[0] == '3' else 0] - keypoints[i + 1, 0 if fdi2[0] == '3' else 1])
            dists[f'{fdi1}-{fdi2}'] = dist.item()
        
        file_name = Path(pred.path).stem
        image_dists[file_name + '_lower'] = dists

    return image_dists


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_dir', type=Path, required=False, default='docs/photo.jpg',
        help='Absolute path to a folder containing intra-oral photographs.',
    )
    parser.add_argument(
        '--out_file', type=Path, required=False, default='measurements.xlsx',
        help='Absolute path to file for saving measurements. Default = measurements.xlsx',
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Option to show model predictions for each input image.'
    )
    args = parser.parse_args()

    automated_dists = automated_measurements(args.in_dir, args.verbose)
    df = pd.DataFrame(automated_dists).T
    df = df.sort_index()
    df.to_excel(args.out_file)
    