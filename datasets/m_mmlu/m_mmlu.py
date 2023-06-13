import os

import datasets
import json

_CITATION = """\
@article{hendryckstest2021,
  title={Measuring Massive Multitask Language Understanding},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
"""

_DESCRIPTION = """\
Measuring Massive Multitask Language Understanding by Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt (ICLR 2021).
"""

LANGS = 'ar,bn,ca,da,de,es,eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh'.split(',')


class MMLUConfig(datasets.BuilderConfig):
    def __init__(self, lang, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.name = 'mmlu_' + lang
        self.test_url = f'datasets/m_mmlu/{lang}_test.json'
        self.dev_url = f'datasets/m_mmlu/{lang}_dev.json'


class MMLU(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        MMLUConfig(lang) for lang in LANGS
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "choices": datasets.features.Sequence(datasets.Value("string")),
                "answer": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage='',
            license='',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": self.config.test_url
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": self.config.dev_url,
                },
            ),

        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            contents = json.load(f)

        for i, instance in enumerate(contents):
            yield i, {
                "question": instance["instruction"],
                "choices": [
                    instance["option_a"],
                    instance["option_b"],
                    instance["option_c"],
                    instance["option_d"],
                ],
                "answer": instance["answer"],
            }
