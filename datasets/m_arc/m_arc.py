"""TODO(arc): Add a description here."""

import json
import os

import datasets
import itertools

# TODO(ai2_arc): BibTeX citation
_CITATION = """\
@article{allenai:arc,
      author    = {Peter Clark  and Isaac Cowhey and Oren Etzioni and Tushar Khot and
                    Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
      title     = {Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
      journal   = {arXiv:1803.05457v1},
      year      = {2018},
}
"""

# TODO(ai2_arc):
_DESCRIPTION = """\
A new dataset of 7,787 genuine grade-school level, multiple-choice science questions, assembled to encourage research in
 advanced question-answering. The dataset is partitioned into a Challenge Set and an Easy Set, where the former contains
 only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm. We are also
 including a corpus of over 14 million science sentences relevant to the task, and an implementation of three neural baseline models for this dataset. We pose ARC as a challenge to the community.
"""


class Ai2ArcConfig(datasets.BuilderConfig):
    """BuilderConfig for Ai2ARC."""

    def __init__(self, lang, **kwargs):
        """BuilderConfig for Ai2Arc.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Ai2ArcConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.name = f'arc_{lang}'
        self.lang = lang



LANGS = 'ar,bn,ca,da,de,es,eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh'.split(',')


class Ai2ArcMultipleChoice(datasets.GeneratorBasedBuilder):
    """TODO(arc): Short description of my dataset."""

    # TODO(arc): Set up version.
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [Ai2ArcConfig(lang) for lang in LANGS]

    def _info(self):
        # TODO(ai2_arc): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "choices": datasets.features.Sequence(datasets.Value("string")),
                    "answerKey": datasets.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://allenai.org/data/arc",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(ai2_arc): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        # dl_dir = dl_manager.download_and_extract(_URL)
        # data_dir = os.path.join(dl_dir, "ARC-V1-Feb2018-2")

        return [

            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": f'datasets/m_arc/{self.config.lang}_train.json'},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": f'datasets/m_arc/{self.config.lang}_validation.json'},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": f'datasets/m_arc/{self.config.lang}_test.json'},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(ai2_arc): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        for i, d in enumerate(data):

            choices = []
            if 'option_a' in d:
                choices.append(d['option_a'])
            if 'option_b' in d:
                choices.append(d['option_b'])
            if 'option_c' in d:
                choices.append(d['option_c'])
            if 'option_d' in d:
                choices.append(d['option_d'])
            if 'option_e' in d:
                choices.append(d['option_e'])

            yield i, {
                'id': d['id'],
                'question': d['instruction'],
                'choices': choices,
                'answerKey': d['answer']
            }
