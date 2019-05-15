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


def compare_result(gt, pred, config):
    if not os.path.exists('gt_sim.json'):
        gt_sim = list()
        for each_ins in gt:
            each_triplet = each_ins['triplet']
            gt_sim.append({
                str((each_triplet[0] + str(each_ins['subject_tid']),
                     each_triplet[1],
                     each_triplet[2] + str(each_ins['object_tid']))): each_ins['duration']
            })
        with open('gt_sim.json', 'w+') as out_gt:
            out_gt.write(json.dumps(gt_sim))

    pred_sim = list()
    for each_ins in pred:
        each_triplet = each_ins['triplet']
        pred_sim.append({
            str(each_triplet): each_ins['duration'],
            'score': each_ins['score']
        })
    with open('pred_{}.json'.format(config), 'w+') as out_pred:
        out_pred.write(json.dumps(pred_sim))


def convert_gt_to_pred(gt, vid):
    gt_res = list()
    for each_ins in gt:
        gt_res.append({
            'triplet': each_ins['triplet'],
            'score': 1.0,
            'duration': each_ins['duration'],
            'sub_traj': each_ins['sub_traj'],
            'obj_traj': each_ins['obj_traj']
        })

    pred_4mat = {
        'results': {
            vid: gt_res
        }
    }
    with open('gt_pred_4mat.json', 'w+') as out_f:
        out_f.write(json.dumps(pred_4mat))


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

    config = '{}_{}_{}'.format(top_tree, overlap, iou_thr)
    prediction_out = 'test_out_{}.json'.format(config)

    with open(prediction_out, 'r') as in_f:
        pred_json = json.load(in_f)

    compare_result(dataset.get_relation_insts(test_vid, no_traj=True), pred_json['results'][test_vid], config)

    # convert_gt_to_pred(dataset.get_relation_insts(test_vid), test_vid)
