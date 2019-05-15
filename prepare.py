import json
import os

root_path = '/home/daivd/PycharmProjects/VidVRD-py3'
st_prediction = os.path.join(root_path, 'baseline/vidvrd-dataset/vidvrd-baseline-output/short-term-predication.json')

with open(st_prediction, 'r') as in_f:
    st = json.load(in_f)

test_vid = 'ILSVRC2015_train_00066007'
with open('test.json', 'w+') as out_f:
    test_out = {
        "results": st[test_vid]
    }
    out_f.write(json.dumps(test_out))
