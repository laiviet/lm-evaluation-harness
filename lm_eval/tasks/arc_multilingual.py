"""
Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
https://arxiv.org/pdf/1803.05457.pdf

The ARC dataset consists of 7,787 science exam questions drawn from a variety
of sources, including science questions provided under license by a research
partner affiliated with AI2. These are text-only, English language exam questions
that span several grade levels as indicated in the files. Each question has a
multiple choice structure (typically 4 answer options). The questions are sorted
into a Challenge Set of 2,590 “hard” questions (those that both a retrieval and
a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.

Homepage: https://allenai.org/data/arc
"""
from lm_eval.base import MultipleChoiceTask

_CITATION = """
@article{Clark2018ThinkYH,
  title={Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
  author={Peter Clark and Isaac Cowhey and Oren Etzioni and Tushar Khot and Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
  journal={ArXiv},
  year={2018},
  volume={abs/1803.05457}
}
"""


class MultilingualARC(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "m_ai2_arc"
    NUM_FEW_SHOT = 25

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        # NOTE:
        out_doc = {
            "id": doc["id"],
            "query": "Question: " + doc["question"] + "\nAnswer:",
            "choices": doc["choices"],
            "gold": ["A", "B", "C", "D", "E"].index(doc["answerKey"]),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]


class arArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'ar_easy'


class arArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'ar_challenge'


class bnArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'bn_easy'


class bnArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'bn_challenge'


class caArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'ca_easy'


class caArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'ca_challenge'


class daArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'da_easy'


class daArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'da_challenge'


class deArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'de_easy'


class deArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'de_challenge'


class esArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'es_easy'


class esArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'es_challenge'


class euArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'eu_easy'


class euArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'eu_challenge'


class frArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'fr_easy'


class frArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'fr_challenge'


class guArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'gu_easy'


class guArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'gu_challenge'


class hiArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'hi_easy'


class hiArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'hi_challenge'


class hrArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'hr_easy'


class hrArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'hr_challenge'


class huArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'hu_easy'


class huArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'hu_challenge'


class hyArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'hy_easy'


class hyArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'hy_challenge'


class idArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'id_easy'


class idArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'id_challenge'


class itArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'it_easy'


class itArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'it_challenge'


class knArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'kn_easy'


class knArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'kn_challenge'


class mlArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'ml_easy'


class mlArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'ml_challenge'


class mrArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'mr_easy'


class mrArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'mr_challenge'


class neArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'ne_easy'


class neArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'ne_challenge'


class nlArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'nl_easy'


class nlArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'nl_challenge'


class ptArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'pt_easy'


class ptArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'pt_challenge'


class roArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'ro_easy'


class roArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'ro_challenge'


class ruArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'ru_easy'


class ruArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'ru_challenge'


class skArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'sk_easy'


class skArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'sk_challenge'


class srArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'sr_easy'


class srArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'sr_challenge'


class svArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'sv_easy'


class svArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'sv_challenge'


class taArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'ta_easy'


class taArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'ta_challenge'


class teArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'te_easy'


class teArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'te_challenge'


class ukArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'uk_easy'


class ukArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'uk_challenge'


class viArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'vi_easy'


class viArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'vi_challenge'


class zhArcEasy(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'zh_easy'


class zhArcChallenge(MultilingualARC):
    DATASET_PATH = 'datasets/m_ai2_arc'
    DATASET_NAME = 'zh_challenge'
