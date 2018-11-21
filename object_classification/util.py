import torch


def ccwh2xyxy(boxs):
    '''
    Args
        boxs: (tensor) boxes in ccwh format, sized [#N, 4]
    Return
        boxs: (tensor) boxes in xyxy format, sized [#N, 4]
    '''
    xy, wh = boxs[:, :2], boxs[:, 2:]
    boxs = torch.cat([xy - wh / 2, xy + wh / 2], dim=1)
    return boxs


def xyxy2ccwh(boxs):
    '''
    Args
        boxs: (tensor) boxes in xyxy format, sized [#N, 4]
    Return
        boxs: (tensor) boxes in ccwh format, sized [#N, 4]
    '''
    wh = boxs[:, 2:] - boxs[:, :2]
    cc = boxs[:, :2] + wh / 2
    boxs = torch.cat([cc, wh], dim=1)
    return boxs


def iou_slow(A, B):
    '''
    Args
        A: (tensor) first set of boxes in ccwh format, sized [N, 4]
        B: (tensor) second set of boxes in ccwh format, sized [M, 4]
    Return
        C: iou[i, j] is the iou of A[i] and B[j], sized [N, M]
    '''
    A = ccwh2xyxy(A)
    B = ccwh2xyxy(B)

    C = torch.zeros(A.size(0), B.size(0))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            inter_x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            inter_y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = inter_x * inter_y
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            union = area_a + area_b - inter
            C[i, j] = inter / union
    return C


def iou(A, B):
    '''
    Args
        A: (tensor) first set of boxes in ccwh format, sized [N, 4]
        B: (tensor) second set of boxes in ccwh format, sized [M, 4]
    Return
        C: iou[i, j] is the iou of A[i] and B[j], sized [N, M]
    '''
    A = ccwh2xyxy(A)
    B = ccwh2xyxy(B)

    N, M = A.size(0), B.size(0)
    A = A.unsqueeze(1).expand(N, M, 4)
    B = B.unsqueeze(0).expand(N, M, 4)

    Ix = torch.min(A[..., 2], B[..., 2]) - torch.max(A[..., 0], B[..., 0])
    Iy = torch.min(A[..., 3], B[..., 3]) - torch.max(A[..., 1], B[..., 1])
    I = Ix.clamp(min=0) * Iy.clamp(min=0)
    Aa = (A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])
    Ab = (B[..., 2] - B[..., 0]) * (B[..., 3] - B[..., 1])
    U = Aa + Ab - I

    return I / U


def to_one_hot(y, n_class):
    y = y.view(-1, 1)
    one_hot = torch.zeros(len(y), n_class, device=y.device)
    one_hot.scatter_(1, y, 1)
    return one_hot


def nms(bboxes, scores, threshold=0.5):
    '''Non maximum suppression.
    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) confidence scores, sized [N,].
      threshold: (float) overlap threshold.
    Returns:
      keep: (tensor) selected indices.
    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (overlap <= threshold)
        if ids.sum() == 0:
            break
        ids = torch.nonzero(ids).squeeze(1)
        order = order[ids + 1]
    return torch.tensor(keep, dtype=torch.long)


class RunningAverage:
    def __init__(self):
        self.iter = 0
        self.avg = 0.0

    def update(self, x):
        self.avg = (self.avg * self.iter + x) / (self.iter + 1)
        self.iter += 1

    def __str__(self):
        if self.iter == 0:
            return 'nan'
        return f'{self.avg:.4f}'
