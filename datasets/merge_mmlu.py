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
    for file_name in tqdm.tqdm(os.listdir(folder_path), desc=folder_path):
        if file_name.endswith(".json"):
            with open(os.path.join(folder_path, file_name)) as f:
                data = json.load(f)['result']
            _, split, i = data['id'].split("/")
            datadict[split].append(data)

    for split, data in datadict.items():
        with open(os.path.join(output_dir, f'{lang}_{split}.json'), "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':

    for code2 in code2_map:
        merge(
            input_dir=f"../translation/mmlu-chatgpt/{code2}",
            output_dir=f"m_mmlu/"
        )
