import torch
import numpy as np
import cv2

import argparse

from edgepoint2 import EdgePoint2Wrapper

def draw_match(img1, img2, pts1, pts2):
    
    def draw_corners(img, corners):
        for i in range(len(corners)):
            start = tuple(corners[i-1][0].astype(int))
            end = tuple(corners[i][0].astype(int))
            cv2.line(img, start, end, (0, 255, 0), 4)
        return img
    
    def put_text(img, num_matches):
        return cv2.putText(img, f'Matches: {num_matches}', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    
    h, w = img1.shape[:2]
    corners_img1 = np.array([[40, 40], [w-41, 40], [w-41, h-41], [40, h-41]], dtype=np.float32).reshape(-1, 1, 2)
    img1 = draw_corners(img1, corners_img1)
    if len(pts1) <= 10 or len(pts2) <= 10:
        return put_text(np.concatenate([img1, img2], axis=1), 0)
    
    H, mask = cv2.findHomography(pts1, pts2, cv2.USAC_MAGSAC, 2, maxIters=10000, confidence=0.999)
    mask = mask.flatten()
    if mask.sum() <= 10:
        return put_text(np.concatenate([img1, img2], axis=1), 0)
    
    corners_img2 = cv2.perspectiveTransform(corners_img1, H)
    img2 = draw_corners(img2, corners_img2)

    img2 = img2.copy()
    img2 = draw_corners(img2, corners_img2)

    pts1 = [cv2.KeyPoint(p[0], p[1], 5) for p in pts1]
    pts2 = [cv2.KeyPoint(p[0], p[1], 5) for p in pts2]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]
    
    img_matches = cv2.drawMatches(img1, pts1, img2, pts2, matches, None,
                                  matchColor=(127, 127, 0), flags=2)
    img_matches = put_text(img_matches, len(matches))

    return img_matches


def match(desc1: torch.Tensor, desc2: torch.Tensor, threshold=-1):
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    
    cossim = torch.einsum("nd,md->nm", desc1, desc2)
    _, match12 = cossim.max(dim=1)
    _, match21 = cossim.max(dim=0)

    idx1 = torch.arange(len(match12), device=match12.device)
    mutual = match21[match12] == idx1

    idx1 = idx1[mutual]
    idx2 = match12[mutual]
    scores = cossim[idx1, idx2]
    
    if threshold > -1:
        mask = scores > threshold
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        scores = scores[mask]
        
    return idx1.cpu().numpy(), idx2.cpu().numpy()


def forward(model, im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    if torch.cuda.is_available():
        im = im.to(torch.device('cuda'))
    with torch.no_grad():
        result = model(im)
        kpts = result[0]['keypoints']
        desc = result[0]['descriptors']
    return kpts.cpu().numpy(), desc


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='camera or video file')
    parser.add_argument('--model', type=str, choices=EdgePoint2Wrapper.cfgs.keys(), default='E64')
    parser.add_argument('--camid', type=int, default=0)
    parser.add_argument('--W', type=int, default=640)
    parser.add_argument('--H', type=int, default=480)
    parser.add_argument('--top_k', type=int, default=4096)
    parser.add_argument('--match_threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    
    ep2 = EdgePoint2Wrapper(args.model, top_k=args.top_k).eval()
    if torch.cuda.is_available():
        ep2 = ep2.cuda()
    
    if args.input == 'camera':
        cap = cv2.VideoCapture(args.camid)
    else:
        cap = cv2.VideoCapture(args.input)
    
    frozen_im = None
    frozen_kpts = None
    frozen_desc = None
    while 1:
        ret, im = cap.read()
        if not ret:
            break
        
        im = cv2.resize(im, (args.W, args.H))
        kpts, desc = forward(ep2, im)
        
        if frozen_im is None:
            frozen_im = im
            frozen_kpts = kpts
            frozen_desc = desc
            continue
        
        idxs1, idxs2 = match(frozen_desc, desc, args.match_threshold)
        matched_im = draw_match(frozen_im, im.copy(), frozen_kpts[idxs1], kpts[idxs2])
        cv2.imshow('matches', matched_im)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):
            frozen_im = im
            frozen_kpts = kpts
            frozen_desc = desc