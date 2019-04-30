import json
from collections import defaultdict

import numpy as np

from relation import VideoRelation, _merge_trajs, _traj_iou_over_common_frames
from trajectory import Trajectory
from tree import TrackTree, TreeNode


def get_dummy_detect(path):
    """
    This is the function to spawn a separate branch with a dummy observation, to account for missing detection.
    :param path: In this version, in missing case, simply duplicate the last frame.
    :return: A separate branch with a dummy observation.
    """


def gating_to_recall_mis_pred(path):
    """
    This is the function to recall the predication which is mistake.
    :param path:
    :return:
    """


if __name__ == '__main__':

