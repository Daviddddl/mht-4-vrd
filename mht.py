import json
from collections import defaultdict

from scipy.spatial.distance import mahalanobis

from tree import TrackTree, TreeNode

from track_utils import check_2_nodes

import numpy as np

import heapq


def origin_mht_relational_association(short_term_relations, truncate_per_segment=3, top_tree=1):
    """
    This is not the very official MHT framework, which mainly is 4 frame-level.
    This func is to associating short-term-relations relational.
    :param top_tree:
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

                # add multi root nodes
                if new_tree_node.duration[0] == 0:
                    # add a new root node
                    track_tree.add(new_tree_node)
                else:
                    # update tree
                    for each_path in tree_paths:
                        if check_2_nodes(each_path[-1], new_tree_node):
                            track_tree.add(new_tree_node, each_path[-1])

                # track Scoring

                # global hypothesis formation

                # track tree pruning

                # generate results
                # print("Begin 2 generate results")
                save_res_path = 'test_out.json'
                top_k_paths, top_k_scores = generate_results(track_tree, save_res_path, top_tree)
                print(top_k_scores)
                print(top_k_paths)


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


def track_score(track_path):
    """
    Scoring the proposal tracklet can b associate 2 st_track possibility.
    Score = weight_motion * score_motion + weight_appearance * score_appearance
    :return:
    """
    weight_motion, score_motion, weight_appearance, score_appearance = 0, 0, 0, 0
    score = weight_motion * score_motion + weight_appearance * score_appearance

    path_score = 0.
    for each_node in track_path:
        path_score += each_node.score
    return path_score / len(track_path)


def global_hypo(track_trees):
    """
    Determine the most likely combination of object tracks at frame k.
    NP-hard, Maximum Weighted Independent Set Problem (MWIS)
    :param track_trees: a set of trees containing all traj hypotheses 4 all targets
    :return:
    """


def generate_results(track_tree, save_res_path, top_k=3):
    """
    :param track_tree:
    :param save_res_path:
    :return:
    """
    path_score_dict = dict()
    for each_path in track_tree.get_paths():
        path_score_dict[track_score(each_path)] = each_path
    sorted_keys = sorted(path_score_dict.keys())
    top_k_res = list()
    for i in range(top_k):
        top_k_res.append(path_score_dict[sorted_keys[i]])
    return top_k_res, sorted_keys[:top_k]


if __name__ == '__main__':
    with open('test.json', 'r') as in_f:
        short_term_relations = json.load(in_f)
    origin_mht_relational_association(short_term_relations)
