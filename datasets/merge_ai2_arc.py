import json
import os
from collections import defaultdict
import tqdm


def load_language_set():
    with open('../..//mInstructLLM/languages.csv') as f:
        lines = f.read().split('\n')
    lines = lines[1:]
    lang_map = dict()
    code2_map = dict()
    order_list = list()
    for line in lines:
        parts = tuple(line.split(','))
        if len(parts) != 4:
            continue
        for p in parts:
            assert p not in lang_map, p
            lang_map[p] = parts
        code2_map[parts[1]] = parts
        order_list.append(parts)
    return code2_map, lang_map, order_list


code2_map, lang_map, order_list = load_language_set()


# Merge json files from a folder
def merge(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(input_dir)
    folder_path = os.path.abspath(input_dir)
    lang = os.path.basename(folder_path)
    # print(base_folder)
    datadict = defaultdict(list)
    available_fields = [
        'instruction',
        'option_a',
        'option_b',
        'option_c',
        'option_d',
        'option_e'
    ]

    for file_name in tqdm.tqdm(os.listdir(folder_path), desc=folder_path):
        if not file_name.endswith(".json"):
            continue

        with open(os.path.join(folder_path, file_name)) as f:
            data = json.load(f)
        result = data['result']
        english_data = data['meta']['data']
        easy_challenge, corpus, i = english_data['id'].split("/")

        if easy_challenge == 'ARC-Easy':
            continue
        answer = 'ABCDE'.index(english_data['answer'])

        fields = [x for x in available_fields if x in english_data]
        if answer >= len(fields) - 1:
            continue

        if not all([x in result for x in fields]):
            continue
        # Make the sample
        translated_data = {
            'id': english_data['id'],
            'answer': english_data['answer'],
        }
        for field in fields:
            translated_data[field] = result[field]
        datadict[corpus].append(translated_data)

    print(f"Writing {lang} to {output_dir}")
    for corpus, data in datadict.items():
        with open(os.path.join(output_dir, f'{lang}_{corpus}.json'), "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(corpus, len(data))


if __name__ == '__main__':

    for code2 in code2_map:
        merge(
            input_dir=f"../translation/ai2_arc-chatgpt/{code2}",
            output_dir=f"m_arc/"
        )
