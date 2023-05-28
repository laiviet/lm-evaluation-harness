# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TruthfulQA dataset."""

import csv
import json

import datasets
from itertools import product

_CITATION = """\
@misc{lin2021truthfulqa,
    title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
    author={Stephanie Lin and Jacob Hilton and Owain Evans},
    year={2021},
    eprint={2109.07958},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
TruthfulQA is a benchmark to measure whether a language model is truthful in
generating answers to questions. The benchmark comprises 817 questions that
span 38 categories, including health, law, finance and politics. Questions are
crafted so that some humans would answer falsely due to a false belief or
misconception. To perform well, models must avoid generating false answers
learned from imitating human texts.
"""

_HOMEPAGE = "https://github.com/sylinrl/TruthfulQA"

_LICENSE = "Apache License 2.0"


class LangTaskConfig(datasets.BuilderConfig):
    """BuilderConfig for TruthfulQA."""

    def __init__(self, lang, task, **kwargs):
        """BuilderConfig for TruthfulQA.
        Args:
          url: *string*, the url to the configuration's data.
          features: *list[string]*, list of features that'll appear in the feature dict.
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.name = f'{lang}_{task}'
        self.url = f"datasets/m_truthful_qa/{self.name}.json"

        assert task in ['mc', 'gen']

        if task == 'mc':
            self.features = datasets.Features(
                {
                    "type": datasets.Value("string"),
                    "category": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "best_answer": datasets.Value("string"),
                    "correct_answers": datasets.features.Sequence(datasets.Value("string")),
                    "incorrect_answers": datasets.features.Sequence(datasets.Value("string")),
                    "source": datasets.Value("string"),
                }
            )
        elif task == 'gen':
            self.features = datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "mc1_targets": {
                        "choices": datasets.features.Sequence(datasets.Value("string")),
                        "labels": datasets.features.Sequence(datasets.Value("int32")),
                    },
                    "mc2_targets": {
                        "choices": datasets.features.Sequence(datasets.Value("string")),
                        "labels": datasets.features.Sequence(datasets.Value("int32")),
                    },
                }
            )


LANGS = 'ar,bn,ca,da,de,es,eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh'.split(',')
TASKS = ['mc', 'gen']


class MultilingualTruthfulQa(datasets.GeneratorBasedBuilder):
    """TruthfulQA is a benchmark to measure whether a language model is truthful in generating answers to questions."""

    BUILDER_CONFIGS = [
        LangTaskConfig(lang, task)
        for lang, task in product(LANGS, TASKS)

    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self.config.features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # data_dir = dl_manager.download_and_extract(self.config.url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": self.config.url
                },
            ),
        ]

    def _generate_examples(self, filepath):
        if "_mc" in self.config.name:
            # Multiple choice data is in a `JSON` file.
            with open(filepath, encoding="utf-8") as f:
                contents = json.load(f)
            for i, row in enumerate(contents):
                print(row)
                yield i, {
                    "question": row["question"],
                    "mc1_targets": {
                        "choices": row["mc1_targets_choices"],
                        "labels": row["mc1_targets_labels"],
                    },
                    "mc2_targets": {
                        "choices": row["mc2_targets_choices"],
                        "labels": row["mc2_targets_labels"],
                    },
                }
        elif "_gen" in self.config.name:
            # Generation is in a `JSON` file.
            with open(filepath, encoding="utf-8") as f:
                contents = json.load(f)
            for i, row in enumerate(contents):
                print(row)

                yield i, {
                    "type": row["type"],
                    "category": row["category"],
                    "question": row["question"],
                    "best_answer": row["best_answer"],
                    "correct_answers": row["correct_answers"],
                    "incorrect_answers": row["incorrect_answers"],
                    "source": row["source"],
                }
        else:
            print('Incorrect config name: ', self.config.name)
