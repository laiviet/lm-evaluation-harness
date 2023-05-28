import glob
import os
import sys
import json
from itertools import product

LANGS = 'ar,bn,ca,da,de,es,eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh'.split(',')

MODELS = [
    'tiny-gpt2'
    'bloom-560',
    'bloom-1b7',
    'bloom-7b1',
]

TASK_TEMPLATES = [
    'arc_{}_easy',
    'arc_{}_challenge',
]


def read_json(json_file):
    with open(json_file, encoding="utf-8") as f:
        contents = json.load(f)
    return contents


all_perfs = dict()

for file in glob.glob('logs/*.json'):
    contents = read_json(file)
    model = contents['config']['model_args'].split('/')[-1]

    results = contents['results']
    for task, perfs in results.items():
        all_perfs[(model, task)] = perfs


for model in MODELS:
    for task_template, lang in product(TASK_TEMPLATES, LANGS):
        task = task_template.format(lang)
        for model in MODELS:
            perf = all_perfs.get((model, task), None)
            if perf is None:
                print(f'  {model}: None')
            else:
                print(f'  {model}: {perf:.2f}'