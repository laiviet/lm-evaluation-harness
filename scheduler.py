LANGS = 'ar,bn,ca,da,de,es,eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh'.split(',')

ARC_EASY_TASKS = [f'arc_{lang}_easy' for lang in LANGS]
ARC_CHALLENGE_TASKS = [f'arc_{lang}_challenge' for lang in LANGS]
ARC_TASKS = ARC_EASY_TASKS + ARC_CHALLENGE_TASKS

TRUTHFULQA_GENERATION_TASKS = [f'truthfulqa_{lang}_gen' for lang in LANGS]
TRUTHFULQA_MULTIPLE_CHOICE_TASKS = [f'truthfulqa_{lang}_mc' for lang in LANGS]
TRUTHFULQA_TASKS = TRUTHFULQA_GENERATION_TASKS + TRUTHFULQA_MULTIPLE_CHOICE_TASKS

model_configs = [
    # ('hf-auto', 'bigscience/bloom-560m', 'bloom-560', 1, 'cuda'),
    ('hf-auto', 'bigscience/bloom-1b7', 'bloom-1b7', 1, 'cuda'),
    # ('hf-auto', 'bigscience/bloom-7b1', 'bloom-7b1', 1, 'cuda'),
    # ('hf-auto', 'sshleifer/tiny-gpt2', 'tiny-gpt2', 1, 'cuda'),
]

commands = []
for model, pretrained, alias, bs, device in model_configs:
    for task in ARC_TASKS:
        cmd = f'python main.py --model {model} ' + \
              f'--model_args pretrained={pretrained} ' + \
              f'--model_alias {alias} ' + \
              f'--tasks {task} ' + \
              f'--batch_size {bs} ' + \
              f'--device {device} '
        commands.append(cmd)
with open('commands.txt', 'a') as f:
    f.write('\n'.join(commands))
    f.write('\n')
