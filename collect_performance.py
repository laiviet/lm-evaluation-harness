import glob
import os
import sys
import json
from itertools import product

LANGS = 'ar,bn,ca,da,de,es,eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh'.split(',')

MODELS = [
    'tiny-gpt2',
    'gpt2',
    'gpt2-medium',
    'gpt2-large',
    'bloom-560m',
    'bloom-1b7',
    'bloom-3b',
    'bloom-7b1',
]

TASK_TEMPLATES = [
    'arc_{}_easy',
    'arc_{}_challenge',
    'truthfulqa_{}_mc',
    'truthfulqa_{}_gen',
    # 'mmlu_{}',
    # 'hellaswag_{}',
]

TASK_METRICS = {
    'arc_{}_easy': 'acc',
    'arc_{}_challenge': 'acc',
    'truthfulqa_{}_gen': 'bleurt',
    # 'truthfulqa_{}_mc': 'bleurt',
}


def read_json(json_file):
    with open(json_file, encoding="utf-8") as f:
        contents = json.load(f)
    return contents


all_perfs = dict()

for file in glob.glob('logs/*.json'):
    # print(file)
    contents = read_json(file)
    model = contents['config']['model_args']
    model = model.split('=')[1].split('/')[-1]

    results = contents['results']
    for task, perfs in results.items():
        all_perfs[(model, task)] = perfs
    # print(model, task)

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
                if 'acc' in perf:
                    m = perf['acc'] * 100.
                elif 'bleurt' in perf:
                    m = perf['bleurt'] * 100.
                print(f'{m:.1f}', end=',')
            else:
                print('-', end=',')
        print('')
    print('')

if __name__ == '__main__':
    pass
