from collections import deque
import numpy as np
import torch

from common import estimate_normals, unproject

# Code adapted from https://github.com/saic-vul/saic_depth_completion

class Statistics(object):
    def __init__(self, maxlen=20):
        self.enum = deque(maxlen=maxlen)
        self.denum = deque(maxlen=maxlen)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.total = 0.0
        self.count = 0
        self.enum.clear()
        self.denum.clear()

    def update(self, value, n):
        self.enum.append(value)
        self.denum.append(n)
        self.count += n
        self.total += value

    @property
    def median(self):
        enum = torch.tensor(list(self.enum))
        denum = torch.tensor(list(self.denum))
        sequence = enum / denum
        return sequence.median().item()

    @property
    def avg(self):
        enum = torch.tensor(list(self.enum))
        denum = torch.tensor(list(self.denum))
        avg = enum.sum() / denum.sum()
        return  avg.item()

    @property
    def global_avg(self):
        return self.total / self.count


class Meter:
    def __init__(self, metric_fn, maxlen=20):
        self.metric_fn = metric_fn
        self.stats = Statistics(maxlen)

    def reset(self):
        self.stats.reset()

    def update(self, pred, gt, mask, data):
        value = self.metric_fn(pred, gt, mask, data)
        if isinstance(value, tuple):
            self.stats.update(value[0].cpu(), value[1])
        else:
            self.stats.update(value.item(), 1)

    @property
    def median(self):
        return self.stats.median
    @property
    def avg(self):
        return self.stats.avg

    @property
    def global_avg(self):
        return self.stats.global_avg

class AggregatedMeter(object):
    def __init__(self, metrics, maxlen=20, delimiter=' # '):
        self.delimiter = delimiter
        self.meters = {
            k: Meter(v, maxlen) for k, v in metrics.items()
        }

    def reset(self):
        for v in self.meters.values():
            v.reset()

    def update(self, pred, gt, mask=None):
        data = {} # Allow metrics to share data
        for v in self.meters.values():
            v.update(pred, gt, mask, data)

    @property
    def suffix(self):
        suffix = []
        for k, v in self.meters.items():
            suffix.append(
                "{}: {:.4f} ({:.4f})".format(k, v.median, v.global_avg)
            )
        return self.delimiter.join(suffix)

class DepthMAE():
    def __init__(self, eps=1e-5, scale=1):
        self.eps = eps
        self.scale = scale

    def __call__(self, pred, gt, mask=None, data=None):
        if mask is None:
            mask = gt > self.eps

        return torch.tensor(np.mean(np.abs(pred[mask] - gt[mask]))) * self.scale

class DepthRMSE():
    def __init__(self, eps=1e-5, scale=1):
        self.eps = eps
        self.scale = scale

    def __call__(self, pred, gt, mask=None, data=None):
        if mask is None:
            mask = gt > self.eps

        return torch.tensor(np.sqrt(np.mean((pred[mask] - gt[mask])**2))) * self.scale

class NormalAngleDifference():
    def __init__(self, K, eps=1e-5):
        self.K = K
        self.eps = eps

    def __call__(self, pred, gt, mask=None, data: dict=None):
        if mask is None:
            mask = gt > self.eps

        
        normals_gt = data.get('normals_gt', torch.from_numpy(-estimate_normals(unproject(gt, K=self.K, depth_is_distance=True))[mask]))
        normals_pred = data.get('normals_pred', torch.from_numpy(-estimate_normals(unproject(pred, K=self.K, depth_is_distance=True))[mask]))

        normals_dot = data.get('normals_dot', torch.bmm(normals_gt.view(-1, 1, 3), normals_pred.view(-1, 3, 1))).clamp(-1, 1)
        
        # Store data
        data['normals_gt'] = normals_gt
        data['normals_pred'] = normals_pred
        data['normals_dot'] = normals_dot

        return torch.rad2deg(torch.acos(normals_dot)).mean()

class NormalAnglePercentage():
    def __init__(self, threshold_angle, K, eps=1e-5):
        self.threshold_angle = threshold_angle
        self.K = K
        self.eps = eps

    def __call__(self, pred, gt, mask=None, data: dict=None):
        if mask is None:
            mask = gt > self.eps
        
        if 'normals_dot' not in data:
            data['normals_gt'] = torch.from_numpy(-estimate_normals(unproject(gt, K=self.K, depth_is_distance=True))[mask])
            data['normals_pred'] = torch.from_numpy(-estimate_normals(unproject(pred, K=self.K, depth_is_distance=True))[mask])
            data['normals_dot'] = torch.bmm(data['normals_gt'].view(-1, 1, 3), data['normals_pred'].view(-1, 3, 1)).clamp(-1, 1)

        normals_angle = torch.rad2deg(torch.acos(data['normals_dot']))

        return (normals_angle < self.threshold_angle).sum() / len(normals_angle) * 100