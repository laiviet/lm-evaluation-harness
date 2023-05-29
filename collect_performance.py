import glob
import os
import sys
import json
from itertools import product

LANGS = 'ar,bn,ca,da,de,es,eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh'.split(',')

MODELS = [
    'tiny-gpt2',
    'bloom-560m',
    'bloom-1b7',
    'bloom-7b1',
]

TASK_TEMPLATES = [
    'arc_{}_easy',
    'arc_{}_challenge',
    'mmlu_{}',
    'truthfulqa_{}',
    'hellaswag_{}',
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
        all_perfs[(model,task)] = perfs

# print(all_perfs)

# print(json.dumps(all_perfs, indent=2))

print('Model', end=',')
print('Task', end=',')
print('Average', end=',')
for lang in LANGS:
    print(lang, end=',')
print('')

for model in MODELS:
    for task_template in TASK_TEMPLATES:
        print(model, end=',')
        print(task_template.replace('_{}', ''), end=',')
        print('-', end=',')
        for lang in LANGS:
            task = task_template.format(lang)
            perf = all_perfs.get((model, task), None)

            if perf:
                acc = perf['acc'] * 100.
                if task.startswith('arc_'):
                    print(f'{acc:.1f}', end=',')
            else:
                print('-', end=',')
        print('')
    print('')


if __name__ == '__main__':
    pass
