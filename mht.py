import json
from collections import defaultdict

import numpy as np

from relation import VideoRelation
from track_utils import check_2_nodes, traj_iou_over_common_frames, merge_trajs
from trajectory import Trajectory
from tree import TrackTree, TreeNode

anno_rpath = 'baseline/vidvrd-dataset'
video_rpath = 'baseline/vidvrd-dataset/videos'
splits = ['train', 'test']
so_id = dict()


def origin_mht_relational_association(short_term_relations,
                                      truncate_per_segment=100, top_tree=10, overlap=0.3, iou_thr=0.3):
    """
    This is not the very official MHT framework, which mainly is 4 frame-level.
    This func is to associating short-term-relations relational.
    :param overlap: overlap 4 obj id, higher, more
    :param iou_thr: iou for associate, higher, less
    :param top_tree:
    :param short_term_relations:
    :param truncate_per_segment:
    :return:
    """
    so_id.clear()
    pstart_relations = defaultdict(list)
    for r in short_term_relations:
        pstart_relations[r['duration'][0]].append(r)

    tree_dict = dict()
    for pstart in sorted(pstart_relations.keys()):
        sorted_relations = sorted(pstart_relations[pstart], key=lambda r: r['score'], reverse=True)
        sorted_relations = sorted_relations[:truncate_per_segment]

        # Traversing truncate_per_segment relations
        for each_rela in sorted_relations:
            subj_label, pred_rela, obj_label = each_rela['triplet']
            score = each_rela['score']
            subj_tracklet = each_rela['sub_traj']
            obj_tracklet = each_rela['obj_traj']
            duration = each_rela['duration']

            subj_traj = Trajectory(duration[0], duration[1], subj_tracklet, score)
            obj_traj = Trajectory(duration[0], duration[1], obj_tracklet, score)

            subj = subj_label + '#' + str(get_obj_id(subj_label, subj_traj, overlap))
            obj = obj_label + '#' + str(get_obj_id(obj_label, obj_traj, overlap))
            each_tree = (subj, obj)

            new_tree_node = TreeNode(name=each_tree,
                                     so_labels=(subj_label, obj_label),
                                     score=score,
                                     st_predicate=pred_rela,
                                     subj_tracklet=subj_tracklet,
                                     obj_tracklet=obj_tracklet,
                                     duration=duration)

            if each_tree not in tree_dict.keys():
                tree_dict[each_tree] = TrackTree()
                tree_dict[each_tree].add(new_tree_node)
            else:
                track_tree = tree_dict[each_tree]
                if duration[0] == 0:
                    track_tree.add(new_tree_node, track_tree.tree)
                else:
                    for each_path in track_tree.get_paths():
                        if check_2_nodes(each_path[-1], new_tree_node, iou_thr):
                            track_tree.add(new_tree_node, each_path[-1])

    # generate results
    pred_set = set()
    video_relation_list = list()
    for each_pair, each_tree in tree_dict.items():
        for each_path in each_tree.get_paths():
            for each_node in each_path:
                pred_set.add(each_node.st_predicate)

        top_k_paths, top_k_scores = generate_results(each_tree, top_tree)
        for each_path in top_k_paths:
            video_relation_list.append(associate_path(each_path))

    print(pred_set)
    return [r.serialize() for r in video_relation_list]


def get_obj_id(obj, obj_traj, overlap_threshold):
    if obj in so_id.keys():
        obj_id = -1
        obj_track_overlap = overlap_threshold
        max_id = -1
        for each_obj_id, each_obj_traj in so_id[obj].items():
            each_overlap = traj_iou_over_common_frames(obj_traj, each_obj_traj)
            max_id = max(max_id, each_obj_id)
            if each_overlap > obj_track_overlap >= overlap_threshold:
                obj_track_overlap = each_overlap
                obj_id = each_obj_id
        if obj_id == -1:
            obj_id = max_id + 1
            so_id[obj][obj_id] = obj_traj
        else:
            so_id[obj][obj_id] = merge_trajs(so_id[obj][obj_id], obj_traj)
    else:
        obj_id = 0
        so_id[obj] = {0: obj_traj}
    return obj_id


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


def generate_results(track_tree, top_k):
    """
    :param top_k:
    :param track_tree:
    :return:
    """
    path_score_dict = dict()
    for each_path in track_tree.get_paths():
        path_score_dict[track_score(each_path)] = each_path
    sorted_keys = sorted(path_score_dict.keys(), reverse=True)
    top_k_res = list()
    if len(sorted_keys) >= top_k:
        for i in range(top_k):
            top_k_res.append(path_score_dict[sorted_keys[i]])
        top_k_scores = sorted_keys[:top_k]
    else:
        for each_key in sorted_keys:
            top_k_res.append(path_score_dict[each_key])
        top_k_scores = sorted_keys
    return top_k_res, top_k_scores


def associate_path(track_path):
    result = None
    preds = dict()
    for each_node in track_path:
        each_st_predicate = each_node.st_predicate
        if each_st_predicate is not None:
            if each_st_predicate in preds.keys():
                preds[each_st_predicate].append(each_node.score)
            else:
                preds[each_st_predicate] = [each_node.score]

    for each_pred, each_scores in preds.items():
        preds[each_pred] = np.mean(each_scores)

    preds_list = sorted(preds.items(), key=lambda item: item[1], reverse=True)
    for each_node in track_path:
        if each_node.duration != [0, 0]:
            sub, obj = each_node.so_labels
            pstart, pend = each_node.duration
            conf_score = each_node.score
            straj = Trajectory(pstart, pend, each_node.subj_tracklet, conf_score)
            otraj = Trajectory(pstart, pend, each_node.obj_tracklet, conf_score)

            for each_pred, pred_score in preds_list:
                if result is None:
                    result = VideoRelation(sub, each_pred, obj, straj, otraj, conf_score + pred_score)
                else:
                    result.extend(straj, otraj, conf_score + pred_score)
    return result


if __name__ == '__main__':
    with open('test2.json', 'r') as test_st_rela_f:
        test_st_rela = json.load(test_st_rela_f)

    result = origin_mht_relational_association(test_st_rela)

    print(len(result))
    # for each_res in result:
    #     print(each_res)
