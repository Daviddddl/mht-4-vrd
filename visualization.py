import json
import os
from dataset.vidvrd import VidVRD
import matplotlib.pyplot as plt


def visualization(data, data_type):
    fontsize = 30
    plt.figure(figsize=(50, 12))
    color = '#8F8FEF' if data_type == 'gt' else '#FAA460'

    x_list = list()
    y_list = list()
    color_list = list()
    for each_ins in data:
        x_list.append(str(each_ins['triplet']))
        y_list.append(tuple(each_ins['duration']))
        color_list.append(color)

    plt.bar(range(len(x_list)), y_list, color=color, tick_label=x_list)
    plt.axis('tight')
    plt.xlim([-1, len(x_list)])
    plt.tight_layout()
    plt.show()


def compare_result(gt, pred, vid):
    gt_sim = list()
    pred_sim = list()
    for each_ins in gt:
        each_triplet = each_ins['triplet']
        gt_sim.append({
            str((each_triplet[0] + str(each_ins['subject_tid']),
                 each_triplet[1],
                 each_triplet[2] + str(each_ins['object_tid']))): each_ins['duration']
        })
    with open('gt_{}.json'.format(vid), 'w+') as out_gt:
        out_gt.write(json.dumps(gt_sim))

    for each_ins in pred:
        each_triplet = each_ins['triplet']
        pred_sim.append({
            str(each_triplet): each_ins['duration'],
            'score': each_ins['score']
        })
    with open('pred_{}.json'.format(vid), 'w+') as out_pred:
        out_pred.write(json.dumps(pred_sim))


if __name__ == '__main__':
    anno_rpath = 'vidvrd-dataset'
    video_rpath = ''
    splits = ['test']
    dataset = VidVRD(anno_rpath=anno_rpath,
                     video_rpath=video_rpath,
                     splits=splits)

    top_tree = 20
    overlap = 0.3
    iou_thr = 0.3

    test_vid = 'ILSVRC2015_train_00066007'

    prediction_out = 'test_out_{}_{}_{}.json'.format(top_tree, overlap, iou_thr)

    with open(prediction_out, 'r') as in_f:
        pred_json = json.load(in_f)

    compare_result(dataset.get_relation_insts(test_vid, no_traj=True), pred_json['results'][test_vid], test_vid)
