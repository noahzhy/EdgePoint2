import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import EdgePoint2


class EdgePoint2Wrapper(nn.Module):

    cfgs = {
        'T32': {'c1': 8, 'c2': 8, 'c3': 16, 'c4': 24, 'cdesc': 32, 'cdetect': 8},
        'T48': {'c1': 8, 'c2': 8, 'c3': 16, 'c4': 24, 'cdesc': 48, 'cdetect': 8},
        'S32': {'c1': 8, 'c2': 8, 'c3': 24, 'c4': 32, 'cdesc': 32, 'cdetect': 8},
        'S48': {'c1': 8, 'c2': 8, 'c3': 24, 'c4': 32, 'cdesc': 48, 'cdetect': 8},
        'S64': {'c1': 8, 'c2': 8, 'c3': 24, 'c4': 32, 'cdesc': 64, 'cdetect': 8},
        'M32': {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 48, 'cdesc': 32, 'cdetect': 8},
        'M48': {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 48, 'cdesc': 48, 'cdetect': 8},
        'M64': {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 48, 'cdesc': 64, 'cdetect': 8},
        'L32': {'c1': 8, 'c2': 16, 'c3': 48, 'c4': 64, 'cdesc': 32, 'cdetect': 8},
        'L48': {'c1': 8, 'c2': 16, 'c3': 48, 'c4': 64, 'cdesc': 48, 'cdetect': 8},
        'L64': {'c1': 8, 'c2': 16, 'c3': 48, 'c4': 64, 'cdesc': 64, 'cdetect': 8},
        'E32': {'c1': 16, 'c2': 16, 'c3': 48, 'c4': 64, 'cdesc': 32, 'cdetect': 16},
        'E48': {'c1': 16, 'c2': 16, 'c3': 48, 'c4': 64, 'cdesc': 48, 'cdetect': 16},
        'E64': {'c1': 16, 'c2': 16, 'c3': 48, 'c4': 64, 'cdesc': 64, 'cdetect': 16},
    }

    def __init__(self, cfg, top_k, k=2, score=0.0):
        super().__init__()
        assert top_k is None or top_k > 0
        self.top_k = top_k
        self.k = k
        self.score_thresh = score

        self.model = EdgePoint2(**self.cfgs[cfg[:3]])
        self.model.load_state_dict(torch.load(f'./weights/{cfg}.pth', 'cpu'))

        self.mp = nn.MaxPool2d(k * 2 + 1, 1, k)

    @torch.inference_mode()
    def forward(self, x):
        assert x.shape[2] % 32 == 0, f"Input height {x.shape[2]} is not divisible by 32"
        assert x.shape[3] % 32 == 0, f"Input width {x.shape[3]} is not divisible by 32"
        _, _, oH, oW = x.shape
        # nH = oH // 32 * 32
        # nW = oW // 32 * 32
        size = torch.tensor([oW, oH], dtype=x.dtype, device=x.device)
        # scale = torch.tensor([oW/nW, oH/nH], dtype=x.dtype, device=x.device)
        # if oW != nW or oH != nH:
        #     x = F.interpolate(x, (nH, nW), mode='bilinear', align_corners=True)

        # raw_desc: (1, C_desc, H', W'), raw_detect: (1, C_detect, H', W')
        raw_desc, raw_detect = self.model(x)

        # Assuming C_detect = 1 based on original code [:, 0]
        scores_map = raw_detect[0, 0] # Shape: (H', W')
        mp_scores = self.mp(raw_detect)[0, 0] # Shape: (H', W')

        # Non-Maximum Suppression (NMS) and Thresholding
        is_peak = (scores_map == mp_scores)
        is_above_thresh = (scores_map > self.score_thresh)
        detect = torch.logical_and(is_peak, is_above_thresh)

        # Apply boundary conditions
        detect[..., :, :4]  = False
        detect[..., :, -4:] = False
        detect[..., :4, :]  = False
        detect[..., -4:, :] = False

        # Get indices (row, col) and scores of detected keypoints
        pts_indices = torch.nonzero(detect, as_tuple=False) # Shape: (N, 2) [row, col]
        # No keypoints detected
        if pts_indices.shape[0] == 0:
            return {
                'keypoints': torch.empty((0, 2), dtype=x.dtype, device=x.device),
                'scores': torch.empty((0,), dtype=x.dtype, device=x.device),
                'descriptors': torch.empty((0, raw_desc.shape[1]), dtype=x.dtype, device=x.device)
            }

        # nonzero returns (row, col) convert to (x, y) -> (col, row)
        kpts = pts_indices[:, [1, 0]].to(x.dtype) # Shape: (N, 2) [x, y]
        scores = scores_map[pts_indices[:, 0], pts_indices[:, 1]] # Shape: (N,)

        # Top-K Selection
        if self.top_k is not None and scores.shape[0] > self.top_k:
            scores, idx = scores.topk(self.top_k)
            kpts = kpts[idx]

        # Descriptor Sampling
        if kpts.shape[0] > 0:
            # Prepare coordinates for sampling: shape (1, N, 1, 2) in range [-1, 1]
            coords = (kpts + 0.5).reshape(1, -1, 1, 2)
            coords = coords / size * 2 - 1 # Normalize to [-1, 1]
            descs = F.grid_sample(raw_desc, coords, mode='bilinear', align_corners=False)
            descs = descs[0, :, :, 0].transpose(-1, -2).contiguous() # Shape: (N, C_desc)
        else:
            descs = raw_desc.new_zeros([0, raw_desc.shape[1]]) # Shape: (0, C_desc)

        return {
            'keypoints': kpts,
            'scores': scores,
            'descriptors': descs
        }


if __name__ == '__main__':
    model_name = 'T32'
    model = EdgePoint2Wrapper(model_name, 2048)
    model.eval()
    # out = model(torch.randn(1, 3, 640, 480))
    # export to as onnx
    torch.onnx.export(
        model,
        torch.randn(1, 3, 640, 480),
        # torch.randn(1, 3, 1280, 960),
        f'onnx/ep2_{model_name}.onnx',
        input_names=['images'],
        output_names=['keypoints', 'scores', 'descriptors'],
        dynamic_axes={
            'keypoints':    {0: 'num_keypoints'},
            'scores':       {0: 'num_keypoints'},
            'descriptors':  {0: 'num_keypoints'}
        },
        do_constant_folding=True,
        opset_version=17
    )
