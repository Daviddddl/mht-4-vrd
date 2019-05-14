import json

st_prediction = 'baseline/vidvrd-dataset/vidvrd-baseline-output/short-term-predication.json'

with open(st_prediction, 'r') as in_f:
    st = json.load(in_f)

test_vid = 'ILSVRC2015_train_00066007'
with open('test.json', 'w+') as out_f:
    out_f.write(json.dumps(st['result'][test_vid]))
