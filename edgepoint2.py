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
        B, _, oH, oW = x.shape
        nH = oH // 32 * 32
        nW = oW // 32 * 32
        size = torch.tensor([nW, nH], dtype=x.dtype, device=x.device)
        scale = torch.tensor([oW/nW, oH/nH], dtype=x.dtype, device=x.device)
        if oW != nW or oH != nH:
            x = F.interpolate(x, (nH, nW), mode='bilinear', align_corners=True)
        
        raw_desc, raw_detect = self.model(x)
        
        detect1 = raw_detect == self.mp(raw_detect)
        detect1[..., :, :4] = False
        detect1[..., :, -4:] = False
        detect1[..., :4, :] = False
        detect1[..., -4:, :] = False
        
        detect2 = raw_detect > self.score_thresh
        detect = torch.logical_and(detect1, detect2)[:,0]
        H = torch.arange(detect.shape[-2], dtype=x.dtype, device=x.device)
        W = torch.arange(detect.shape[-1], dtype=x.dtype, device=x.device)
        H, W = torch.meshgrid(H, W)
        ind = torch.stack([W, H], dim=-1)
        kpts = [ind[detect[b]] for b in range(B)]
        scores = [raw_detect[b,0,detect[b]] for b in range(B)]
        
        if self.top_k is not None:
            for i in range(B):
                score, idx = scores[i].topk(min(self.top_k, scores[i].shape[0]))
                scores[i] = score
                kpts[i] = kpts[i][idx]
        
        descs = [self.model.sample(raw_desc[b:b+1], (kpts[b] + 0.5).reshape(1, -1, 1, 2) / size * 2 - 1)[0,:,:,0].mT.contiguous() if kpts[b].shape[0] > 0 else raw_desc.new_zeros([0, raw_desc.shape[1]]) for b in range(B)]
        
        return [  
				   {'keypoints': kpts[b] * scale,
					'scores': scores[b],
					'descriptors': descs[b]} for b in range(B) 
			   ]


if __name__ == '__main__':
    im = torch.randn(1, 3, 640, 480)
    model = EdgePoint2Wrapper('T32', 1000, 2, -5)