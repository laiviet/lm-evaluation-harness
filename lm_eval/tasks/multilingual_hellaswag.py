"""
HellaSwag: Can a Machine Really Finish Your Sentence?
https://arxiv.org/pdf/1905.07830.pdf

Hellaswag is a commonsense inference challenge dataset. Though its questions are
trivial for humans (>95% accuracy), state-of-the-art models struggle (<48%). This is
achieved via Adversarial Filtering (AF), a data collection paradigm wherein a
series of discriminators iteratively select an adversarial set of machine-generated
wrong answers. AF proves to be surprisingly robust. The key insight is to scale up
the length and complexity of the dataset examples towards a critical 'Goldilocks'
zone wherein generated text is ridiculous to humans, yet often misclassified by
state-of-the-art models.

Homepage: https://rowanzellers.com/hellaswag/
"""
import re
from lm_eval.base import MultipleChoiceTask

_CITATION = """
@inproceedings{zellers2019hellaswag,
    title={HellaSwag: Can a Machine Really Finish Your Sentence?},
    author={Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
    booktitle ={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    year={2019}
}
"""


class HellaSwag(MultipleChoiceTask):
    VERSION = 1
    DATASET_PATH = "datasets/m_hellaswag"
    NUM_FEW_SHOT = 10

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": self.preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [self.preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    @classmethod
    def preprocess(cls, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]


LANGS = 'ar,bn,ca,da,de,es,eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh'.split(',')


class deHellaSwag(HellaSwag):
    DATASET_NAME = "de"


class arHellaSwag(HellaSwag):
    DATASET_NAME = "ar"


class bnHellaSwag(HellaSwag):
    DATASET_NAME = "bn"


class caHellaSwag(HellaSwag):
    DATASET_NAME = "ca"


class daHellaSwag(HellaSwag):
    DATASET_NAME = "da"


class esHellaSwag(HellaSwag):
    DATASET_NAME = "es"


class euHellaSwag(HellaSwag):
    DATASET_NAME = "eu"


class frHellaSwag(HellaSwag):
    DATASET_NAME = "fr"


class guHellaSwag(HellaSwag):
    DATASET_NAME = "gu"


class hiHellaSwag(HellaSwag):
    DATASET_NAME = "hi"


class hrHellaSwag(HellaSwag):
    DATASET_NAME = "hr"


class huHellaSwag(HellaSwag):
    DATASET_NAME = "hu"


class hyHellaSwag(HellaSwag):
    DATASET_NAME = "hy"


class idHellaSwag(HellaSwag):
    DATASET_NAME = "id"


class itHellaSwag(HellaSwag):
    DATASET_NAME = "it"


class knHellaSwag(HellaSwag):
    DATASET_NAME = "kn"


class mlHellaSwag(HellaSwag):
    DATASET_NAME = "ml"


class mrHellaSwag(HellaSwag):
    DATASET_NAME = "mr"


class neHellaSwag(HellaSwag):
    DATASET_NAME = "ne"


class nlHellaSwag(HellaSwag):
    DATASET_NAME = "nl"


class ptHellaSwag(HellaSwag):
    DATASET_NAME = "pt"


class roHellaSwag(HellaSwag):
    DATASET_NAME = "ro"


class ruHellaSwag(HellaSwag):
    DATASET_NAME = "ru"


class skHellaSwag(HellaSwag):
    DATASET_NAME = "sk"


class srHellaSwag(HellaSwag):
    DATASET_NAME = "sr"


class svHellaSwag(HellaSwag):
    DATASET_NAME = "sv"


class taHellaSwag(HellaSwag):
    DATASET_NAME = "ta"


class teHellaSwag(HellaSwag):
    DATASET_NAME = "te"


class ukHellaSwag(HellaSwag):
    DATASET_NAME = "uk"


class viHellaSwag(HellaSwag):
    DATASET_NAME = "vi"


class zhHellaSwag(HellaSwag):
    DATASET_NAME = "zh"
