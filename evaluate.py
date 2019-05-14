import json

from dataset import VidVRD
from evaluation import eval_video_object, eval_action, eval_visual_relation


def evaluate_object(dataset, split, prediction):
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_object_insts(vid)
    mean_ap, ap_class = eval_video_object(groundtruth, prediction)


def evaluate_action(dataset, split, prediction):
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_action_insts(vid)
    mean_ap, ap_class = eval_action(groundtruth, prediction)


def evaluate_relation(dataset, split, prediction):
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_relation_insts(vid)
    mean_ap, rec_at_n, mprec_at_n = eval_visual_relation(groundtruth, prediction)

    # evaluate in zero-shot setting, if u need
    print('-- zero-shot setting')
    zeroshot_triplets = dataset.get_triplets(split).difference(
        dataset.get_triplets('train'))
    groundtruth = dict()
    zs_prediction = dict()
    for vid in dataset.get_index(split):
        gt_relations = dataset.get_relation_insts(vid)
        zs_gt_relations = []
        for r in gt_relations:
            if tuple(r['triplet']) in zeroshot_triplets:
                zs_gt_relations.append(r)
        if len(zs_gt_relations) > 0:
            groundtruth[vid] = zs_gt_relations
            zs_prediction[vid] = []
            for r in prediction[vid]:
                if tuple(r['triplet']) in zeroshot_triplets:
                    zs_prediction[vid].append(r)
    mean_ap, rec_at_n, mprec_at_n = eval_visual_relation(groundtruth, zs_prediction)


if __name__ == '__main__':
    anno_rpath = 'vidvrd-dataset'
    video_rpath = ''
    splits = ['test']

    top_tree = 20
    overlap = 0.2
    iou_thr = 0.2

    prediction = 'test_out.json'
    dataset = VidVRD(anno_rpath=anno_rpath,
                     video_rpath=video_rpath,
                     splits=splits)

    print('Loading prediction from {}'.format(prediction))
    with open(prediction, 'r') as fin:
        pred = json.load(fin)

    print('Number of videos in prediction: {}'.format(len(pred['results'])))

    # modify the split ['train', 'test']
    evaluate_relation(dataset, 'test', pred['results'])
