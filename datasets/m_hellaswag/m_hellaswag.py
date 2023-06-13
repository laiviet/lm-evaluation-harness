"""TODO(hellaswag): Add a description here."""

import json

import datasets

# TODO(hellaswag): BibTeX citation
_CITATION = """\
@inproceedings{zellers2019hellaswag,
    title={HellaSwag: Can a Machine Really Finish Your Sentence?},
    author={Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
    booktitle ={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    year={2019}
}
"""

_DESCRIPTION = """
HellaSwag: Can a Machine Really Finish Your Sentence? is a new dataset for commonsense NLI. A paper was published at ACL2019.
"""


class HellaswagConfig(datasets.BuilderConfig):

    def __init__(self, lang, **kwargs):
        """BuilderConfig for Hellaswag.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(HellaswagConfig, self).__init__(**kwargs)
        self.name = f'hellaswag_{lang}'
        self.url = f"datasets/m_hellaswag/{lang}_validation.json"

LANGS = 'ar,bn,ca,da,de,es,eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh'.split(',')



class Hellaswag(datasets.GeneratorBasedBuilder):
    """TODO(hellaswag): Short description of my dataset."""

    # TODO(hellaswag): Set up version.
    VERSION = datasets.Version("0.1.0")

    BUILDER_CONFIGS = [
        HellaswagConfig(lang)
        for lang in LANGS
    ]

    def _info(self):
        # TODO(hellaswag): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    # These are the features of your dataset like images, labels ...
                    "ind": datasets.Value("int32"),
                    "activity_label": datasets.Value("string"),
                    "ctx_a": datasets.Value("string"),
                    "ctx_b": datasets.Value("string"),
                    "ctx": datasets.Value("string"),
                    "endings": datasets.features.Sequence(datasets.Value("string")),
                    "source_id": datasets.Value("string"),
                    "split": datasets.Value("string"),
                    "split_type": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://rowanzellers.com/hellaswag/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(hellaswag): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": self.config.url},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(hellaswag): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            contents = json.load(f)
        print('Loaded', len(contents), 'examples')
        for i, data in enumerate(contents):
            yield i, {
                "ind": int(data["ind"]),
                "activity_label": data["activity_label"],
                "ctx_a": data['ctx_a'],
                "ctx_b": data['ctx_b'],
                "ctx": data["ctx"],
                "endings": data["endings"],
                "source_id": data["source_id"],
                "split": data["split"],
                "split_type": data["split_type"],
                "label": data['label'],
            }