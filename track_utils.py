import itertools
from collections import deque
from copy import deepcopy

import numpy as np
from math import sin, cos, pi, sqrt
from trajectory import Trajectory

LARGE = 10000


def check_2_nodes(tree_tail, new_tree_node):
    tail_start_f, tail_end_f = tree_tail.duration
    new_node_start_f, new_node_end_f = new_tree_node.duration
    if tail_start_f < new_node_start_f < tail_end_f < new_node_end_f:
        overlap_start, overlap_end = new_node_start_f, tail_end_f
        subj_tail_track = tree_tail.subj_tracklet[(overlap_start - tail_start_f):
                                                  (overlap_end - tail_start_f)]
        obj_tail_track = tree_tail.obj_tracklet[(overlap_start - tail_start_f):
                                                (overlap_end - tail_start_f)]
        subj_new_track = new_tree_node.subj_tracklet[(overlap_start - new_node_start_f):
                                                     (overlap_end - new_node_start_f)]
        obj_new_track = new_tree_node.obj_tracklet[(overlap_start - new_node_start_f):
                                                   (overlap_end - new_node_start_f)]
        # generate trajectory
        subj_tail_traj = Trajectory(overlap_start, overlap_end, subj_tail_track, tree_tail.score)
        subj_new_traj = Trajectory(overlap_start, overlap_end, subj_new_track, new_tree_node.score)
        obj_tail_traj = Trajectory(overlap_start, overlap_end, obj_tail_track, tree_tail.score)
        obj_new_traj = Trajectory(overlap_start, overlap_end, obj_new_track, new_tree_node.score)

        return check_overlap(subj_tail_traj, subj_new_traj) and check_overlap(obj_tail_traj, obj_new_traj)


def check_overlap(traj1, traj2, iou_thr=0.5):
    return traj_iou_over_common_frames(traj1, traj2) >= iou_thr


def traj_iou_over_common_frames(traj_1, traj_2):
    if traj_1.pend <= traj_2.pstart or traj_2.pend <= traj_1.pstart:  # no overlap
        return 0
    if traj_1.pstart <= traj_2.pstart:
        t1 = deepcopy(traj_1)
        t2 = deepcopy(traj_2)
    else:
        t1 = deepcopy(traj_2)
        t2 = deepcopy(traj_1)
    overlap_length = t1.pend - t2.pstart
    t1.rois = deque(itertools.islice(t1.rois, t2.pstart - t1.pstart, t1.pend - t1.pstart))
    t2.rois = deque(itertools.islice(t2.rois, 0, t1.pend - t2.pstart))
    iou = traj_iou([t1], [t2])
    return iou[0, 0]


def traj_iou(trajs1, trajs2):
    """
    Compute the pairwise trajectory IoU in trajs1 and trajs2.
    Assumuing all trajectories in trajs1 and trajs2 start at same frame and
    end at same frame.
    """
    bboxes1 = np.asarray([[[roi.left(), roi.top(), roi.right(), roi.bottom()]
                           for roi in traj.rois] for traj in trajs1])
    if id(trajs1) == id(trajs2):
        bboxes2 = bboxes1
    else:
        bboxes2 = np.asarray([[[roi.left(), roi.top(), roi.right(), roi.bottom()]
                               for roi in traj.rois] for traj in trajs2])
    iou = cubic_iou(bboxes1, bboxes2)
    return iou


def cubic_iou(bboxes1, bboxes2):
    # bboxes: n x t x 4 (left, top, right, bottom)
    if id(bboxes1) == id(bboxes2):
        bboxes1 = bboxes1.transpose((1, 0, 2))
        bboxes2 = bboxes1
    else:
        bboxes1 = bboxes1.transpose((1, 0, 2))
        bboxes2 = bboxes2.transpose((1, 0, 2))
    # compute cubic-IoU
    # bboxes: t x n x 4
    iou = _intersect(bboxes1, bboxes2)
    union = _union(bboxes1, bboxes2)
    np.subtract(union, iou, out=union)
    np.divide(iou, union, out=iou)
    return iou


def _intersect(bboxes1, bboxes2):
    """
    bboxes: t x n x 4
    """
    assert bboxes1.shape[0] == bboxes2.shape[0]
    t = bboxes1.shape[0]
    inters = np.zeros((bboxes1.shape[1], bboxes2.shape[1]), dtype=np.float32)
    _min = np.empty((bboxes1.shape[1], bboxes2.shape[1]), dtype=np.float32)
    _max = np.empty((bboxes1.shape[1], bboxes2.shape[1]), dtype=np.float32)
    w = np.empty((bboxes1.shape[1], bboxes2.shape[1]), dtype=np.float32)
    h = np.empty((bboxes1.shape[1], bboxes2.shape[1]), dtype=np.float32)
    for i in range(t):
        np.maximum.outer(bboxes1[i, :, 0], bboxes2[i, :, 0], out=_min)
        np.minimum.outer(bboxes1[i, :, 2], bboxes2[i, :, 2], out=_max)
        np.subtract(_max + 1, _min, out=w)
        w.clip(min=0, out=w)
        np.maximum.outer(bboxes1[i, :, 1], bboxes2[i, :, 1], out=_min)
        np.minimum.outer(bboxes1[i, :, 3], bboxes2[i, :, 3], out=_max)
        np.subtract(_max + 1, _min, out=h)
        h.clip(min=0, out=h)
        np.multiply(w, h, out=w)
        inters += w
    return inters


def _union(bboxes1, bboxes2):
    if id(bboxes1) == id(bboxes2):
        w = bboxes1[:, :, 2] - bboxes1[:, :, 0] + 1
        h = bboxes1[:, :, 3] - bboxes1[:, :, 1] + 1
        area = np.sum(w * h, axis=0)
        unions = np.add.outer(area, area)
    else:
        w = bboxes1[:, :, 2] - bboxes1[:, :, 0] + 1
        h = bboxes1[:, :, 3] - bboxes1[:, :, 1] + 1
        area1 = np.sum(w * h, axis=0)
        w = bboxes2[:, :, 2] - bboxes2[:, :, 0] + 1
        h = bboxes2[:, :, 3] - bboxes2[:, :, 1] + 1
        area2 = np.sum(w * h, axis=0)
        unions = np.add.outer(area1, area2)
    return unions


class PrioItem:
    """Item storable in PriorityQueue."""

    def __init__(self, prio, data):
        """Init."""
        self.prio = prio
        self.data = data

    def __lt__(self, b):
        """lt comparison."""
        return self.prio < b.prio


def anyitem(iterable):
    """Retrieve 'first' item from set."""
    try:
        return next(iter(iterable))
    except StopIteration:
        return None


def connected_components(connections):
    """Get all connected components."""
    seen = set()

    def component(node):
        nodes = {node}
        while nodes:
            node = nodes.pop()
            seen.add(node)
            nodes |= connections[node] - seen
            yield node

    for node in list(connections.keys()):
        if node not in seen:
            yield set(component(node))


def overlap(a, b):
    """Check if boundingboxes overlap."""
    return (a[1] >= b[0] and a[0] <= b[1] and
            a[3] >= b[2] and a[2] <= b[3])


def overlap_pa(a, b):
    """Return percentage of bbox a being in b."""
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0])) \
                   * max(0, min(a[3], b[3]) - max(a[2], b[2]))
    aa = (a[1] - a[0]) * (a[3] - a[2])
    return intersection / aa


def eigsorted(cov):
    """Return eigenvalues, sorted."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def cov_ellipse(cov, nstd):
    """Get the covariance ellipse."""
    vals, vecs = eigsorted(cov)
    r1, r2 = nstd * np.sqrt(vals)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    return r1, r2, theta


def gaussian_bbox(x, P, nstd=2):
    """Return boudningbox for gaussian."""
    r1, r2, theta = cov_ellipse(P, nstd)
    ux = r1 * cos(theta)
    uy = r1 * sin(theta)
    vx = r2 * cos(theta + pi / 2)
    vy = r2 * sin(theta + pi / 2)

    dx = sqrt(ux * ux + vx * vx)
    dy = sqrt(uy * uy + vy * vy)

    return (float(x[0] - dx),
            float(x[0] + dx),
            float(x[1] - dy),
            float(x[1] + dy))


def within(p, bbox):
    """Check if point is within bbox."""
    return (bbox[0] <= p[0] <= bbox[1]) and (bbox[2] <= p[1] <= bbox[3])
