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


if __name__ == '__main__':
    root_path = '/home/daivd/PycharmProjects/VidVRD-py3'
    anno_rpath = os.path.join(root_path, 'baseline/vidvrd-dataset')
    video_rpath = os.path.join(root_path, 'baseline/vidvrd-dataset/videos')
    splits = ['train', 'test']
    dataset = VidVRD(anno_rpath, video_rpath, splits)

    vid = 'ILSVRC2015_train_00010018'

    top_tree = 20
    overlap = 0.2
    iou_thr = 0.2
    test_result_name = 'mht_test_relation_prediction_v4_{}_{}_{}.json'.format(top_tree, overlap, iou_thr)

    with open(test_result_name, 'r') as in_f:
        pred_json = json.load(in_f)

    visualization(dataset.get_relation_insts(vid), 'gt')
