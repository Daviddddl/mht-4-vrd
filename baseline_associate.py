import json
from collections import defaultdict
from collections import deque
from copy import deepcopy
import itertools
import os
import pickle

from baseline.relation import VideoRelation
from baseline.trajectory import Trajectory
from baseline.trajectory import traj_iou
from baseline.track_tree import TrackTree, TreeNode


def greedy_relational_association(short_term_relations, truncate_per_segment=100):
    # group short-term relations by their staring frames
    pstart_relations = defaultdict(list)
    for r in short_term_relations:
        pstart_relations[r['duration'][0]].append(r)

    video_relation_list = []
    last_modify_rel_list = []
    for pstart in sorted(pstart_relations.keys()):
        last_modify_rel_list.sort(key=lambda r: r.score(), reverse=True)
        sorted_relations = sorted(pstart_relations[pstart], key=lambda r: r['score'], reverse=True)
        sorted_relations = sorted_relations[:truncate_per_segment]

        cur_modify_rel_list = []
        for rel in sorted_relations:
            conf_score = rel['score']
            sub, pred, obj = rel['triplet']
            _, pend = rel['duration']
            straj = Trajectory(pstart, pend, rel['sub_traj'])
            otraj = Trajectory(pstart, pend, rel['obj_traj'])

            for r in last_modify_rel_list:
                if r.triplet() == tuple(rel['triplet']) and r.both_overlap(straj, otraj, iou_thr=0.5):
                    # merge
                    r.extend(straj, otraj, conf_score)
                    last_modify_rel_list.remove(r)
                    cur_modify_rel_list.append(r)
                    break
            else:
                r = VideoRelation(sub, pred, obj, straj, otraj, conf_score)
                video_relation_list.append(r)
                cur_modify_rel_list.append(r)

        last_modify_rel_list = cur_modify_rel_list

    return [r.serialize() for r in video_relation_list]


def origin_mht_relational_association(short_term_relations, truncate_per_segment=3, top_tree=3):
    """
    This is not the very official MHT framework, which mainly is 4 frame-level.
    This func is to associating short-term-relations relational.
    ECCV 2018 update this original MHT 2 RNNs-Gating Network.
    :param short_term_relations:
    :param truncate_per_segment:
    :return:
    """

    # targets_trees_path = 'targets_trees.pkl'
    # triplet_with_id_path = 'triplet_with_id.json'
    # if os.path.exists(targets_trees_path) and os.path.exists(triplet_with_id_path):
    #     with open(targets_trees_path, 'rb') as in_f:
    #         targets_trees = pickle.load(in_f)
    #     with open(triplet_with_id_path, 'r') as in_f:
    #         triplet_with_id = json.load(in_f)['triplet_with_id']
    # else:
        # Construct & update track tree
    pstart_relations = defaultdict(list)
    for r in short_term_relations:
        pstart_relations[r['duration'][0]].append(r)

    targets_trees = dict()
    triplet_with_id = list()

    for pstart in sorted(pstart_relations.keys()):
        sorted_relations = sorted(pstart_relations[pstart], key=lambda r: r['score'], reverse=True)
        sorted_relations = sorted_relations[:truncate_per_segment]

        # Traversing truncate_per_segment relations
        for each_rela in sorted_relations:
            subj, pred_rela, obj = each_rela['triplet']
            score = each_rela['score']
            duration = each_rela['duration']

            for target in [subj, obj]:
                if target == subj:
                    tracklet = each_rela['sub_traj']
                else:
                    tracklet = each_rela['obj_traj']

                if target in targets_trees.keys():
                    # get all of same label trees
                    current_num = 0
                    for each_tree_name in targets_trees[target].keys():
                        label, id = each_tree_name.split('#')
                        current_num = max(current_num, int(id))
                    # figure out whether update the tree or create a new one
                    update_trees = generate_update_trees(tree_dict=targets_trees[target],
                                                         tracklet=tracklet,
                                                         duration=duration)
                    if len(update_trees) > 0:
                        # update trees
                        tree_name = target + '#' + str(current_num)
                        for each_update_tree in update_trees:
                            update_node = TreeNode(
                                name=tree_name,
                                duration=duration,
                                score=score,
                                tracklet=tracklet
                            )
                            each_update_tree.pruning_track_tree()
                            each_update_tree.add(update_node, trackal=True)

                    else:
                        # create a new tree
                        tree_name = target + '#' + str(current_num + 1)
                        target_tree = TrackTree(tree_root_name=tree_name,
                                                score=score,
                                                tracklet=tracklet,
                                                duration=duration)
                        targets_trees[target][tree_name] = target_tree

                else:
                    # create a new tree
                    tree_name = target + '#0'
                    target_tree = TrackTree(tree_root_name=tree_name,
                                            score=score,
                                            tracklet=tracklet,
                                            duration=duration)
                    targets_trees[target] = {
                        tree_name: target_tree
                    }

                if target == subj:
                    subj = tree_name
                if target == obj:
                    obj = tree_name
            triplet = [subj, pred_rela, obj]
            triplet_with_id.append(triplet)

        # with open('targets_trees.pkl', 'wb+') as out_f:
        #     pickle.dump(targets_trees, out_f)
        # with open('triplet_with_id', 'w+') as out_f:
        #     out_f.write(json.dumps({'triplet_with_id': triplet_with_id}))

    # Global_hypo

    # Pruning_track_tree

    # Generate_results

    results = []
    for each_triplet in triplet_with_id:
        subj, rela, obj = each_triplet
        sub_target = subj.split('#')[0]
        obj_target = obj.split('#')[0]
        subj_tree = targets_trees[sub_target][subj]
        subj_trajs = subj_tree.generate_traj()
        obj_tree = targets_trees[obj_target][obj]
        obj_trajs = obj_tree.generate_traj()

        if len(subj_trajs) < top_tree:
            top_tree_subj = len(subj_trajs)
        else:
            top_tree_subj = top_tree
        if len(obj_trajs) < top_tree:
            top_tree_obj = len(obj_trajs)
        else:
            top_tree_obj = len(obj_trajs)

        for idsubj in range(top_tree_subj):
            subj_traj = subj_trajs[idsubj]
            subj_start = subj_traj.pstart
            subj_end = subj_traj.pend
            subj_length = subj_end - subj_start
            subj_score = subj_traj.score
            for idobj in range(top_tree_obj):
                obj_traj = obj_trajs[idobj]
                obj_start = obj_traj.pstart
                obj_end = obj_traj.pend
                obj_length = obj_end - obj_start
                obj_score = obj_traj.score
                duration = [max(subj_start, obj_start), min(subj_end, obj_end)]
                subj_bbox = []
                obj_bbox = []
                for i in range(duration[0] - subj_start, duration[1] - subj_start):
                    subj_bbox.append(subj_traj.bbox_at(i))
                for i in range(duration[0] - subj_start, duration[1] - obj_start):
                    obj_bbox.append(obj_traj.bbox_at(i))
                straj = Trajectory(duration[0] - subj_start, duration[1] - subj_start, subj_bbox)
                otraj = Trajectory(duration[0] - subj_start, duration[1] - obj_start, obj_bbox)

                instance = VideoRelation(sub_target, rela, obj_target, straj, otraj, (subj_score + obj_score) / (subj_length + obj_length))
                results.append(instance)

    results = [r.serialize() for r in results]
    # with open('test_out.json', 'w+') as out_f:
    #     out_f.write(json.dumps(results))
    # print(len(results))
    results.sort(key=lambda r: r['score'], reverse=True)
    for each_result in results:
        print(each_result['triplet'],each_result['duration'], each_result['score'])
    return results


def generate_update_trees(tree_dict, tracklet, duration):
    update_tree_set = set()
    fstart, fend = duration

    if fend - fstart == 30:
        fmid = fstart + 15
        traj1 = Trajectory(fstart, fmid, tracklet[0: 15])
        traj2 = Trajectory(fmid, fend, tracklet[15: 30])
    else:
        traj1 = Trajectory(fstart, fend, tracklet[0: 15])
        traj2 = None

    for each_tree in tree_dict.keys():
        # figure it out, this step need 2 use gating & track_score
        # iou
        for each_traj in tree_dict[each_tree].generate_traj():
            pstart = each_traj.pstart
            pend = each_traj.pend
            rois = []
            for i in range(pend - 15, pend):
                # print(list(each_traj.bbox_at(i)))
                rois.append(list(each_traj.bbox_at(i)))
            each_traj_tail = Trajectory(pend - 15, pend, rois)
            if not check_overlap(traj1, each_traj_tail):
                update_tree_set.add(tree_dict[each_tree])
            if traj2 is not None:
                if not check_overlap(traj2, each_traj_tail):
                    update_tree_set.add(tree_dict[each_tree])
    return update_tree_set


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


def gating(st_traj, distance_threshold):
    """
    Where the next observation of the track is expected to appear. Use mahalanobis distance.
    :param st_track: track 2 b predicted
    :param distance_threshold: distance threshold
    :return: a gating area where the next observation of the track is expected 2 appear.
    """
    from baseline.mahalanobis import tf_maha
    location = st_traj[-1]
    predict = st_traj.predict
    mahalanobis_distance = tf_maha(location, predict)
    if mahalanobis_distance  <= distance_threshold:
        return True
    return False


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


if __name__ == '__main__':
    # rpath = '/home/daivd/PycharmProjects/VidVRD-py3/'
    #
    # short_term_relations_path = rpath + 'baseline/vidvrd-dataset/vidvrd-baseline-output/short-term-predication.json'

    # with open(short_term_relations_path, 'r') as st_rela_in:
    #     short_term_relations = json.load(st_rela_in)

    # result = greedy_relational_association(short_term_relations['ILSVRC2015_train_00010001'])

    with open('test.json', 'r') as test_st_rela_f:
        test_st_rela = json.load(test_st_rela_f)

    result = origin_mht_relational_association(test_st_rela)
    # origin_mht_relational_association(test_st_rela)

    # print(len(result))
    # print(result[0].keys())
