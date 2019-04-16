from collections import defaultdict

from scipy.spatial.distance import mahalanobis

from tree import TrackTree, TreeNode

from track_utils import check_2_nodes

import numpy as np


def origin_mht_relational_association(short_term_relations, truncate_per_segment=3, top_tree=3):
    """
    This is not the very official MHT framework, which mainly is 4 frame-level.
    This func is to associating short-term-relations relational.
    :param short_term_relations:
    :param truncate_per_segment:
    :return:
    """
    pstart_relations = defaultdict(list)
    for r in short_term_relations:
        pstart_relations[r['duration'][0]].append(r)

    tree_dict = dict()

    for pstart in sorted(pstart_relations.keys()):
        sorted_relations = sorted(pstart_relations[pstart], key=lambda r: r['score'], reverse=True)
        sorted_relations = sorted_relations[:truncate_per_segment]

        # Traversing truncate_per_segment relations
        for each_rela in sorted_relations:
            subj, pred_rela, obj = each_rela['triplet']
            score = each_rela['score']
            subj_tracklet = each_rela['sub_traj']
            obj_tracklet = each_rela['obj_traj']
            duration = each_rela['duration']
            each_triplet = (subj, pred_rela, obj)  # This is tree name

            if each_triplet not in tree_dict.keys():
                tree_dict[each_triplet] = TrackTree(tree_root_triplet=each_triplet,
                                                    score=score,
                                                    subj_tracklet=subj_tracklet,
                                                    obj_tracklet=obj_tracklet,
                                                    duration=duration)
            else:
                track_tree = tree_dict[each_triplet]
                new_tree_node = TreeNode(triplet=each_triplet,
                                         score=score,
                                         subj_tracklet=subj_tracklet,
                                         obj_tracklet=obj_tracklet,
                                         duration=duration)
                tree_paths = track_tree.get_paths()

                # subj_gating = get_gating(new_tree_node.subj_tracklet)
                # obj_gating = get_gating(new_tree_node.obj_tracklet)

                # update tree
                for each_path in tree_paths:
                    if check_2_nodes(each_path[-1], new_tree_node):
                        track_tree.add(new_tree_node, each_path[-1])
                    else:
                        track_tree.add()
                    # track Scoring

                # global hypothesis formation

                # track tree pruning


def get_gating(pre_traj, distance_threshold=0.5):
    """
    Where the next observation of the track is expected to appear. Use mahalanobis distance.
    :param pre_traj: track 2 b predicted
    :param distance_threshold: distance threshold
    :return: a gating area where the next observation of the track is expected 2 appear.
    """
    pre_location = np.array(pre_traj[-1])
    predict = np.array(gating_predict(pre_traj))
    distance = np.dot((predict - pre_location).T, )
    mahalanobis_distance = mahalanobis(pre_location, predict, pre_traj)
    if mahalanobis_distance <= distance_threshold:
        return True
    return False


def gating_predict(pre_traj):
    predict_result = pre_traj[-1]
    return predict_result


def track_score(weight_motion, score_motion, weight_appearance, score_appearance):
    """
    Scoring the proposal tracklet can b associate 2 st_track possibility.
    Score = weight_motion * score_motion + weight_appearance * score_appearance
    :param weight_motion:
    :param score_motion:
    :param weight_appearance:
    :param score_appearance:
    :return:
    """

    return weight_motion * score_motion + weight_appearance * score_appearance


def global_hypo(track_trees):
    """
    Determine the most likely combination of object tracks at frame k.
    NP-hard, Maximum Weighted Independent Set Problem (MWIS)
    :param track_trees: a set of trees containing all traj hypotheses 4 all targets
    :return:
    """


def generate_results(track_trees):
    """
    Generate traj results finally.
    :param track_trees:
    :return: trajs
    """
