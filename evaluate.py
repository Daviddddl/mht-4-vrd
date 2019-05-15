import json
import os
from dataset import VidVRD
from evaluation import eval_video_object, eval_action, eval_visual_relation
from mht import origin_mht_relational_association
from visualization import compare_result


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


def evaluate_relation(dataset, split, prediction, zshot=False):
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_relation_insts(vid)
    mean_ap, rec_at_n, mprec_at_n = eval_visual_relation(groundtruth, prediction)

    if zshot:
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
    dataset = VidVRD(anno_rpath=anno_rpath,
                     video_rpath=video_rpath,
                     splits=splits)

    top_tree = 20
    overlap = 0.3
    iou_thr = 0.8

    test_vid = 'ILSVRC2015_train_00066007'
    config = '{}_{}_{}'.format(top_tree, overlap, iou_thr)

    prediction_out = 'test_out_{}.json'.format(config)

    if os.path.exists(prediction_out):
        print('Loading prediction from {}'.format(prediction_out))
        with open(prediction_out, 'r') as fin:
            result = json.load(fin)
    else:
        with open('test.json', 'r') as test_st_rela_f:
            test_st_rela = json.load(test_st_rela_f)

        result = origin_mht_relational_association(test_st_rela['results'])

        with open(prediction_out, 'w+') as out_f:
            result = {
                "results": {test_vid: result}
            }
            out_f.write(json.dumps(result))

    print('Number of videos in prediction: {}'.format(len(result['results'])))

    # prediction_out = 'gt_pred_4mat.json'
    # with open(prediction_out, 'r') as gt_f:
    #     result = json.load(gt_f)

    evaluate_relation(dataset, 'test', result['results'])

    compare_result(dataset.get_relation_insts(test_vid, no_traj=True), result['results'][test_vid], config)
