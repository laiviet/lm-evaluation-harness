from pprint import pprint
from typing import List, Union

import sacrebleu
import lm_eval.base

from . import superglue
from . import glue
from . import arc
from . import coqa
from . import race
from . import webqs
from . import anli
from . import wsc273
from . import winogrande
from . import quac
from . import hellaswag
from . import swag
from . import openbookqa
from . import squad
from . import naturalqs
from . import sat
from . import arithmetic
from . import lambada
from . import piqa
from . import prost
from . import mc_taco
from . import triviaqa
from . import pubmedqa
from . import sciq
from . import qasper
from . import qa4mre
from . import translation
from . import headqa
from . import mathqa
from . import hendrycks_ethics
from . import drop
from . import unscramble
from . import logiqa
from . import hendrycks_test
from . import hendrycks_math
from . import cbt
from . import lambada_cloze
from . import pile
from . import wikitext
from . import lambada_multilingual
from . import mutual
from . import truthfulqa
from . import blimp
from . import asdiv
from . import gsm8k
from . import storycloze
from . import toxigen
from . import crowspairs
from . import json
from . import xcopa
from . import bigbench
from . import xstorycloze
from . import xwinograd
from . import pawsx
from . import xnli
from . import mgsm
from . import truthfulqa_multilingual
from . import arc_multilingual

########################################
# Translation tasks
########################################

# 6 total
gpt3_translation_benchmarks = {
    "wmt14": ["en-fr", "fr-en"],  # French
    "wmt16": ["en-ro", "ro-en", "de-en", "en-de"],  # German, Romanian
}


# 28 total
selected_translation_benchmarks = {
    **gpt3_translation_benchmarks,
    "wmt20": sacrebleu.get_langpairs_for_testset("wmt20"),
    "iwslt17": ["en-ar", "ar-en"],  # Arabic
}

# 319 total
all_translation_benchmarks = {
    ts: sacrebleu.get_langpairs_for_testset(ts)
    for ts in sacrebleu.get_available_testsets()
}


########################################
# All tasks
########################################

LANGS = 'ar,bn,ca,da,de,es,eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh'.split(',')


TASK_REGISTRY = {
    # ARC Multilingual
    "arc_ar_easy": arc_multilingual.arArcEasy,
    "arc_ar_challenge": arc_multilingual.arArcChallenge,
    "arc_bn_easy": arc_multilingual.bnArcEasy,
    "arc_bn_challenge": arc_multilingual.bnArcChallenge,
    "arc_ca_easy": arc_multilingual.caArcEasy,
    "arc_ca_challenge": arc_multilingual.caArcChallenge,
    "arc_da_easy": arc_multilingual.daArcEasy,
    "arc_da_challenge": arc_multilingual.daArcChallenge,
    "arc_de_easy": arc_multilingual.deArcEasy,
    "arc_de_challenge": arc_multilingual.deArcChallenge,
    "arc_es_easy": arc_multilingual.esArcEasy,
    "arc_es_challenge": arc_multilingual.esArcChallenge,
    "arc_eu_easy": arc_multilingual.euArcEasy,
    "arc_eu_challenge": arc_multilingual.euArcChallenge,
    "arc_fr_easy": arc_multilingual.frArcEasy,
    "arc_fr_challenge": arc_multilingual.frArcChallenge,
    "arc_gu_easy": arc_multilingual.guArcEasy,
    "arc_gu_challenge": arc_multilingual.guArcChallenge,
    "arc_hi_easy": arc_multilingual.hiArcEasy,
    "arc_hi_challenge": arc_multilingual.hiArcChallenge,
    "arc_hr_easy": arc_multilingual.hrArcEasy,
    "arc_hr_challenge": arc_multilingual.hrArcChallenge,
    "arc_hu_easy": arc_multilingual.huArcEasy,
    "arc_hu_challenge": arc_multilingual.huArcChallenge,
    "arc_hy_easy": arc_multilingual.hyArcEasy,
    "arc_hy_challenge": arc_multilingual.hyArcChallenge,
    "arc_id_easy": arc_multilingual.idArcEasy,
    "arc_id_challenge": arc_multilingual.idArcChallenge,
    "arc_it_easy": arc_multilingual.itArcEasy,
    "arc_it_challenge": arc_multilingual.itArcChallenge,
    "arc_kn_easy": arc_multilingual.knArcEasy,
    "arc_kn_challenge": arc_multilingual.knArcChallenge,
    "arc_ml_easy": arc_multilingual.mlArcEasy,
    "arc_ml_challenge": arc_multilingual.mlArcChallenge,
    "arc_mr_easy": arc_multilingual.mrArcEasy,
    "arc_mr_challenge": arc_multilingual.mrArcChallenge,
    "arc_ne_easy": arc_multilingual.neArcEasy,
    "arc_ne_challenge": arc_multilingual.neArcChallenge,
    "arc_nl_easy": arc_multilingual.nlArcEasy,
    "arc_nl_challenge": arc_multilingual.nlArcChallenge,
    "arc_pt_easy": arc_multilingual.ptArcEasy,
    "arc_pt_challenge": arc_multilingual.ptArcChallenge,
    "arc_ro_easy": arc_multilingual.roArcEasy,
    "arc_ro_challenge": arc_multilingual.roArcChallenge,
    "arc_ru_easy": arc_multilingual.ruArcEasy,
    "arc_ru_challenge": arc_multilingual.ruArcChallenge,
    "arc_sk_easy": arc_multilingual.skArcEasy,
    "arc_sk_challenge": arc_multilingual.skArcChallenge,
    "arc_sr_easy": arc_multilingual.srArcEasy,
    "arc_sr_challenge": arc_multilingual.srArcChallenge,
    "arc_sv_easy": arc_multilingual.svArcEasy,
    "arc_sv_challenge": arc_multilingual.svArcChallenge,
    "arc_ta_easy": arc_multilingual.taArcEasy,
    "arc_ta_challenge": arc_multilingual.taArcChallenge,
    "arc_te_easy": arc_multilingual.teArcEasy,
    "arc_te_challenge": arc_multilingual.teArcChallenge,
    "arc_uk_easy": arc_multilingual.ukArcEasy,
    "arc_uk_challenge": arc_multilingual.ukArcChallenge,
    "arc_vi_easy": arc_multilingual.viArcEasy,
    "arc_vi_challenge": arc_multilingual.viArcChallenge,
    "arc_zh_easy": arc_multilingual.zhArcEasy,
    "arc_zh_challenge": arc_multilingual.zhArcChallenge,
    # GLUE
    "cola": glue.CoLA,
    "mnli": glue.MNLI,
    "mnli_mismatched": glue.MNLIMismatched,
    "mrpc": glue.MRPC,
    "rte": glue.RTE,
    "qnli": glue.QNLI,
    "qqp": glue.QQP,
    # "stsb": glue.STSB, # not implemented yet
    "sst": glue.SST,
    "wnli": glue.WNLI,
    # SuperGLUE
    "boolq": superglue.BoolQ,
    "cb": superglue.CommitmentBank,
    "copa": superglue.Copa,
    "multirc": superglue.MultiRC,
    "record": superglue.ReCoRD,
    "wic": superglue.WordsInContext,
    "wsc": superglue.SGWinogradSchemaChallenge,
    # Order by benchmark/genre?
    "coqa": coqa.CoQA,
    "drop": drop.DROP,
    "lambada_openai": lambada.LambadaOpenAI,
    "lambada_standard": lambada.LambadaStandard,
    "lambada_openai_cloze": lambada_cloze.LambadaOpenAICloze,
    "lambada_standard_cloze": lambada_cloze.LambadaStandardCloze,
    # multilingual lambada
    **lambada_multilingual.construct_tasks(),
    "wikitext": wikitext.WikiText,
    # "cbt-cn": cbt.CBTCN, # disabled pending context length fix
    # "cbt-ne": cbt.CBTNE, # disabled pending context length fix
    "piqa": piqa.PiQA,
    "prost": prost.PROST,
    "mc_taco": mc_taco.MCTACO,
    # Science related
    "pubmedqa": pubmedqa.Pubmed_QA,
    "sciq": sciq.SciQ,
    "qasper": qasper.QASPER,
    "qa4mre_2011": qa4mre.QA4MRE_2011,
    "qa4mre_2012": qa4mre.QA4MRE_2012,
    "qa4mre_2013": qa4mre.QA4MRE_2013,
    "triviaqa": triviaqa.TriviaQA,
    "arc_easy": arc.ARCEasy,
    "arc_challenge": arc.ARCChallenge,
    # "quac": quac.QuAC, # not implemented yet
    "logiqa": logiqa.LogiQA,
    "hellaswag": hellaswag.HellaSwag,
    "swag": swag.SWAG,
    "openbookqa": openbookqa.OpenBookQA,
    "squad2": squad.SQuAD2,
    "race": race.RACE,
    # "naturalqs": naturalqs.NaturalQs, # not implemented yet
    "headqa": headqa.HeadQAEsDeprecated,  # for backwards compat - headqa used to default to es
    "headqa_es": headqa.HeadQAEs,
    "headqa_en": headqa.HeadQAEn,
    "mathqa": mathqa.MathQA,
    "webqs": webqs.WebQs,
    "wsc273": wsc273.WinogradSchemaChallenge273,
    "winogrande": winogrande.Winogrande,
    "anli_r1": anli.ANLIRound1,
    "anli_r2": anli.ANLIRound2,
    "anli_r3": anli.ANLIRound3,
    "ethics_cm": hendrycks_ethics.EthicsCM,
    "ethics_deontology": hendrycks_ethics.EthicsDeontology,
    "ethics_justice": hendrycks_ethics.EthicsJustice,
    "ethics_utilitarianism_original": hendrycks_ethics.EthicsUtilitarianismOriginal,
    "ethics_utilitarianism": hendrycks_ethics.EthicsUtilitarianism,
    "ethics_virtue": hendrycks_ethics.EthicsVirtue,
    "truthfulqa_mc": truthfulqa.TruthfulQAMultipleChoice,
    "truthfulqa_gen": truthfulqa.TruthfulQAGeneration,
    "truthfulqa_ar_mc": truthfulqa_multilingual.arTruthfulQAMultipleChoice,
    "truthfulqa_ar_gen": truthfulqa_multilingual.arTruthfulQAGeneration,
    "truthfulqa_bn_mc": truthfulqa_multilingual.bnTruthfulQAMultipleChoice,
    "truthfulqa_bn_gen": truthfulqa_multilingual.bnTruthfulQAGeneration,
    "truthfulqa_ca_mc": truthfulqa_multilingual.caTruthfulQAMultipleChoice,
    "truthfulqa_ca_gen": truthfulqa_multilingual.caTruthfulQAGeneration,
    "truthfulqa_da_mc": truthfulqa_multilingual.daTruthfulQAMultipleChoice,
    "truthfulqa_da_gen": truthfulqa_multilingual.daTruthfulQAGeneration,
    "truthfulqa_de_mc": truthfulqa_multilingual.deTruthfulQAMultipleChoice,
    "truthfulqa_de_gen": truthfulqa_multilingual.deTruthfulQAGeneration,
    "truthfulqa_es_mc": truthfulqa_multilingual.esTruthfulQAMultipleChoice,
    "truthfulqa_es_gen": truthfulqa_multilingual.esTruthfulQAGeneration,
    "truthfulqa_eu_mc": truthfulqa_multilingual.euTruthfulQAMultipleChoice,
    "truthfulqa_eu_gen": truthfulqa_multilingual.euTruthfulQAGeneration,
    "truthfulqa_fr_mc": truthfulqa_multilingual.frTruthfulQAMultipleChoice,
    "truthfulqa_fr_gen": truthfulqa_multilingual.frTruthfulQAGeneration,
    "truthfulqa_gu_mc": truthfulqa_multilingual.guTruthfulQAMultipleChoice,
    "truthfulqa_gu_gen": truthfulqa_multilingual.guTruthfulQAGeneration,
    "truthfulqa_hi_mc": truthfulqa_multilingual.hiTruthfulQAMultipleChoice,
    "truthfulqa_hi_gen": truthfulqa_multilingual.hiTruthfulQAGeneration,
    "truthfulqa_hr_mc": truthfulqa_multilingual.hrTruthfulQAMultipleChoice,
    "truthfulqa_hr_gen": truthfulqa_multilingual.hrTruthfulQAGeneration,
    "truthfulqa_hu_mc": truthfulqa_multilingual.huTruthfulQAMultipleChoice,
    "truthfulqa_hu_gen": truthfulqa_multilingual.huTruthfulQAGeneration,
    "truthfulqa_hy_mc": truthfulqa_multilingual.hyTruthfulQAMultipleChoice,
    "truthfulqa_hy_gen": truthfulqa_multilingual.hyTruthfulQAGeneration,
    "truthfulqa_id_mc": truthfulqa_multilingual.idTruthfulQAMultipleChoice,
    "truthfulqa_id_gen": truthfulqa_multilingual.idTruthfulQAGeneration,
    "truthfulqa_it_mc": truthfulqa_multilingual.itTruthfulQAMultipleChoice,
    "truthfulqa_it_gen": truthfulqa_multilingual.itTruthfulQAGeneration,
    "truthfulqa_kn_mc": truthfulqa_multilingual.knTruthfulQAMultipleChoice,
    "truthfulqa_kn_gen": truthfulqa_multilingual.knTruthfulQAGeneration,
    "truthfulqa_ml_mc": truthfulqa_multilingual.mlTruthfulQAMultipleChoice,
    "truthfulqa_ml_gen": truthfulqa_multilingual.mlTruthfulQAGeneration,
    "truthfulqa_mr_mc": truthfulqa_multilingual.mrTruthfulQAMultipleChoice,
    "truthfulqa_mr_gen": truthfulqa_multilingual.mrTruthfulQAGeneration,
    "truthfulqa_ne_mc": truthfulqa_multilingual.neTruthfulQAMultipleChoice,
    "truthfulqa_ne_gen": truthfulqa_multilingual.neTruthfulQAGeneration,
    "truthfulqa_nl_mc": truthfulqa_multilingual.nlTruthfulQAMultipleChoice,
    "truthfulqa_nl_gen": truthfulqa_multilingual.nlTruthfulQAGeneration,
    "truthfulqa_pt_mc": truthfulqa_multilingual.ptTruthfulQAMultipleChoice,
    "truthfulqa_pt_gen": truthfulqa_multilingual.ptTruthfulQAGeneration,
    "truthfulqa_ro_mc": truthfulqa_multilingual.roTruthfulQAMultipleChoice,
    "truthfulqa_ro_gen": truthfulqa_multilingual.roTruthfulQAGeneration,
    "truthfulqa_ru_mc": truthfulqa_multilingual.ruTruthfulQAMultipleChoice,
    "truthfulqa_ru_gen": truthfulqa_multilingual.ruTruthfulQAGeneration,
    "truthfulqa_sv_mc": truthfulqa_multilingual.svTruthfulQAMultipleChoice,
    "truthfulqa_sv_gen": truthfulqa_multilingual.svTruthfulQAGeneration,
    "truthfulqa_ta_mc": truthfulqa_multilingual.taTruthfulQAMultipleChoice,
    "truthfulqa_ta_gen": truthfulqa_multilingual.taTruthfulQAGeneration,
    "truthfulqa_te_mc": truthfulqa_multilingual.teTruthfulQAMultipleChoice,
    "truthfulqa_te_gen": truthfulqa_multilingual.teTruthfulQAGeneration,
    "truthfulqa_uk_mc": truthfulqa_multilingual.ukTruthfulQAMultipleChoice,
    "truthfulqa_uk_gen": truthfulqa_multilingual.ukTruthfulQAGeneration,
    "truthfulqa_vi_mc": truthfulqa_multilingual.viTruthfulQAMultipleChoice,
    "truthfulqa_vi_gen": truthfulqa_multilingual.viTruthfulQAGeneration,
    "truthfulqa_zh_mc": truthfulqa_multilingual.zhTruthfulQAMultipleChoice,
    "truthfulqa_zh_gen": truthfulqa_multilingual.zhTruthfulQAGeneration,

    # dialogue
    "mutual": mutual.MuTual,
    "mutual_plus": mutual.MuTualPlus,
    # math
    "math_algebra": hendrycks_math.MathAlgebra,
    "math_counting_and_prob": hendrycks_math.MathCountingAndProbability,
    "math_geometry": hendrycks_math.MathGeometry,
    "math_intermediate_algebra": hendrycks_math.MathIntermediateAlgebra,
    "math_num_theory": hendrycks_math.MathNumberTheory,
    "math_prealgebra": hendrycks_math.MathPrealgebra,
    "math_precalc": hendrycks_math.MathPrecalculus,
    "math_asdiv": asdiv.Asdiv,
    "gsm8k": gsm8k.GradeSchoolMath8K,
    # arithmetic
    "arithmetic_2da": arithmetic.Arithmetic2DPlus,
    "arithmetic_2ds": arithmetic.Arithmetic2DMinus,
    "arithmetic_3da": arithmetic.Arithmetic3DPlus,
    "arithmetic_3ds": arithmetic.Arithmetic3DMinus,
    "arithmetic_4da": arithmetic.Arithmetic4DPlus,
    "arithmetic_4ds": arithmetic.Arithmetic4DMinus,
    "arithmetic_5da": arithmetic.Arithmetic5DPlus,
    "arithmetic_5ds": arithmetic.Arithmetic5DMinus,
    "arithmetic_2dm": arithmetic.Arithmetic2DMultiplication,
    "arithmetic_1dc": arithmetic.Arithmetic1DComposite,
    # TODO Perhaps make these groups of tasks
    #   e.g. anli, arithmetic, openai_translations, harness_translations
    # hendrycksTest (57 tasks)
    **hendrycks_test.create_all_tasks(),
    # e.g. wmt14-fr-en
    **translation.create_tasks_from_benchmarks(gpt3_translation_benchmarks),
    # chef's selection, mostly wmt20
    **translation.create_tasks_from_benchmarks(selected_translation_benchmarks),
    # Word Scrambling and Manipulation Tasks
    "anagrams1": unscramble.Anagrams1,
    "anagrams2": unscramble.Anagrams2,
    "cycle_letters": unscramble.CycleLetters,
    "random_insertion": unscramble.RandomInsertion,
    "reversed_words": unscramble.ReversedWords,
    # Pile
    "pile_arxiv": pile.PileArxiv,
    "pile_books3": pile.PileBooks3,
    "pile_bookcorpus2": pile.PileBookCorpus2,
    "pile_dm-mathematics": pile.PileDmMathematics,
    "pile_enron": pile.PileEnron,
    "pile_europarl": pile.PileEuroparl,
    "pile_freelaw": pile.PileFreeLaw,
    "pile_github": pile.PileGithub,
    "pile_gutenberg": pile.PileGutenberg,
    "pile_hackernews": pile.PileHackernews,
    "pile_nih-exporter": pile.PileNIHExporter,
    "pile_opensubtitles": pile.PileOpenSubtitles,
    "pile_openwebtext2": pile.PileOpenWebText2,
    "pile_philpapers": pile.PilePhilPapers,
    "pile_pile-cc": pile.PilePileCc,
    "pile_pubmed-abstracts": pile.PilePubmedAbstracts,
    "pile_pubmed-central": pile.PilePubmedCentral,
    "pile_stackexchange": pile.PileStackExchange,
    "pile_uspto": pile.PileUspto,
    "pile_ubuntu-irc": pile.PileUbuntuIrc,
    "pile_wikipedia": pile.PileWikipedia,
    "pile_youtubesubtitles": pile.PileYoutubeSubtitles,
    # BLiMP
    "blimp_adjunct_island": blimp.BlimpAdjunctIsland,
    "blimp_anaphor_gender_agreement": blimp.BlimpAnaphorGenderAgreement,
    "blimp_anaphor_number_agreement": blimp.BlimpAnaphorNumberAgreement,
    "blimp_animate_subject_passive": blimp.BlimpAnimateSubjectPassive,
    "blimp_animate_subject_trans": blimp.BlimpAnimateSubjectTrans,
    "blimp_causative": blimp.BlimpCausative,
    "blimp_complex_NP_island": blimp.BlimpComplex_NPIsland,
    "blimp_coordinate_structure_constraint_complex_left_branch": blimp.BlimpCoordinateStructureConstraintComplexLeftBranch,
    "blimp_coordinate_structure_constraint_object_extraction": blimp.BlimpCoordinateStructureConstraintObjectExtraction,
    "blimp_determiner_noun_agreement_1": blimp.BlimpDeterminerNounAgreement_1,
    "blimp_determiner_noun_agreement_2": blimp.BlimpDeterminerNounAgreement_2,
    "blimp_determiner_noun_agreement_irregular_1": blimp.BlimpDeterminerNounAgreementIrregular_1,
    "blimp_determiner_noun_agreement_irregular_2": blimp.BlimpDeterminerNounAgreementIrregular_2,
    "blimp_determiner_noun_agreement_with_adj_2": blimp.BlimpDeterminerNounAgreementWithAdj_2,
    "blimp_determiner_noun_agreement_with_adj_irregular_1": blimp.BlimpDeterminerNounAgreementWithAdjIrregular_1,
    "blimp_determiner_noun_agreement_with_adj_irregular_2": blimp.BlimpDeterminerNounAgreementWithAdjIrregular_2,
    "blimp_determiner_noun_agreement_with_adjective_1": blimp.BlimpDeterminerNounAgreementWithAdjective_1,
    "blimp_distractor_agreement_relational_noun": blimp.BlimpDistractorAgreementRelationalNoun,
    "blimp_distractor_agreement_relative_clause": blimp.BlimpDistractorAgreementRelativeClause,
    "blimp_drop_argument": blimp.BlimpDropArgument,
    "blimp_ellipsis_n_bar_1": blimp.BlimpEllipsisNBar_1,
    "blimp_ellipsis_n_bar_2": blimp.BlimpEllipsisNBar_2,
    "blimp_existential_there_object_raising": blimp.BlimpExistentialThereObjectRaising,
    "blimp_existential_there_quantifiers_1": blimp.BlimpExistentialThereQuantifiers_1,
    "blimp_existential_there_quantifiers_2": blimp.BlimpExistentialThereQuantifiers_2,
    "blimp_existential_there_subject_raising": blimp.BlimpExistentialThereSubjectRaising,
    "blimp_expletive_it_object_raising": blimp.BlimpExpletiveItObjectRaising,
    "blimp_inchoative": blimp.BlimpInchoative,
    "blimp_intransitive": blimp.BlimpIntransitive,
    "blimp_irregular_past_participle_adjectives": blimp.BlimpIrregularPastParticipleAdjectives,
    "blimp_irregular_past_participle_verbs": blimp.BlimpIrregularPastParticipleVerbs,
    "blimp_irregular_plural_subject_verb_agreement_1": blimp.BlimpIrregularPluralSubjectVerbAgreement_1,
    "blimp_irregular_plural_subject_verb_agreement_2": blimp.BlimpIrregularPluralSubjectVerbAgreement_2,
    "blimp_left_branch_island_echo_question": blimp.BlimpLeftBranchIslandEchoQuestion,
    "blimp_left_branch_island_simple_question": blimp.BlimpLeftBranchIslandSimpleQuestion,
    "blimp_matrix_question_npi_licensor_present": blimp.BlimpMatrixQuestionNpiLicensorPresent,
    "blimp_npi_present_1": blimp.BlimpNpiPresent_1,
    "blimp_npi_present_2": blimp.BlimpNpiPresent_2,
    "blimp_only_npi_licensor_present": blimp.BlimpOnlyNpiLicensorPresent,
    "blimp_only_npi_scope": blimp.BlimpOnlyNpiScope,
    "blimp_passive_1": blimp.BlimpPassive_1,
    "blimp_passive_2": blimp.BlimpPassive_2,
    "blimp_principle_A_c_command": blimp.BlimpPrinciple_ACCommand,
    "blimp_principle_A_case_1": blimp.BlimpPrinciple_ACase_1,
    "blimp_principle_A_case_2": blimp.BlimpPrinciple_ACase_2,
    "blimp_principle_A_domain_1": blimp.BlimpPrinciple_ADomain_1,
    "blimp_principle_A_domain_2": blimp.BlimpPrinciple_ADomain_2,
    "blimp_principle_A_domain_3": blimp.BlimpPrinciple_ADomain_3,
    "blimp_principle_A_reconstruction": blimp.BlimpPrinciple_AReconstruction,
    "blimp_regular_plural_subject_verb_agreement_1": blimp.BlimpRegularPluralSubjectVerbAgreement_1,
    "blimp_regular_plural_subject_verb_agreement_2": blimp.BlimpRegularPluralSubjectVerbAgreement_2,
    "blimp_sentential_negation_npi_licensor_present": blimp.BlimpSententialNegationNpiLicensorPresent,
    "blimp_sentential_negation_npi_scope": blimp.BlimpSententialNegationNpiScope,
    "blimp_sentential_subject_island": blimp.BlimpSententialSubjectIsland,
    "blimp_superlative_quantifiers_1": blimp.BlimpSuperlativeQuantifiers_1,
    "blimp_superlative_quantifiers_2": blimp.BlimpSuperlativeQuantifiers_2,
    "blimp_tough_vs_raising_1": blimp.BlimpToughVsRaising_1,
    "blimp_tough_vs_raising_2": blimp.BlimpToughVsRaising_2,
    "blimp_transitive": blimp.BlimpTransitive,
    "blimp_wh_island": blimp.BlimpWhIsland,
    "blimp_wh_questions_object_gap": blimp.BlimpWhQuestionsObjectGap,
    "blimp_wh_questions_subject_gap": blimp.BlimpWhQuestionsSubjectGap,
    "blimp_wh_questions_subject_gap_long_distance": blimp.BlimpWhQuestionsSubjectGapLongDistance,
    "blimp_wh_vs_that_no_gap": blimp.BlimpWhVsThatNoGap,
    "blimp_wh_vs_that_no_gap_long_distance": blimp.BlimpWhVsThatNoGapLongDistance,
    "blimp_wh_vs_that_with_gap": blimp.BlimpWhVsThatWithGap,
    "blimp_wh_vs_that_with_gap_long_distance": blimp.BlimpWhVsThatWithGapLongDistance,
    "toxigen": toxigen.ToxiGen,
    "crows_pairs_english": crowspairs.CrowsPairsEnglish,
    "crows_pairs_english_race_color": crowspairs.CrowsPairsEnglishRaceColor,
    "crows_pairs_english_socioeconomic": crowspairs.CrowsPairsEnglishSocioeconomic,
    "crows_pairs_english_gender": crowspairs.CrowsPairsEnglishGender,
    "crows_pairs_english_age": crowspairs.CrowsPairsEnglishAge,
    "crows_pairs_english_religion": crowspairs.CrowsPairsEnglishReligion,
    "crows_pairs_english_disability": crowspairs.CrowsPairsEnglishDisability,
    "crows_pairs_english_sexual_orientation": crowspairs.CrowsPairsEnglishSexualOrientation,
    "crows_pairs_english_nationality": crowspairs.CrowsPairsEnglishNationality,
    "crows_pairs_english_physical_appearance": crowspairs.CrowsPairsEnglishPhysicalAppearance,
    "crows_pairs_english_autre": crowspairs.CrowsPairsEnglishAutre,
    "crows_pairs_french": crowspairs.CrowsPairsFrench,
    "crows_pairs_french_race_color": crowspairs.CrowsPairsFrenchRaceColor,
    "crows_pairs_french_socioeconomic": crowspairs.CrowsPairsFrenchSocioeconomic,
    "crows_pairs_french_gender": crowspairs.CrowsPairsFrenchGender,
    "crows_pairs_french_age": crowspairs.CrowsPairsFrenchAge,
    "crows_pairs_french_religion": crowspairs.CrowsPairsFrenchReligion,
    "crows_pairs_french_disability": crowspairs.CrowsPairsFrenchDisability,
    "crows_pairs_french_sexual_orientation": crowspairs.CrowsPairsFrenchSexualOrientation,
    "crows_pairs_french_nationality": crowspairs.CrowsPairsFrenchNationality,
    "crows_pairs_french_physical_appearance": crowspairs.CrowsPairsFrenchPhysicalAppearance,
    "crows_pairs_french_autre": crowspairs.CrowsPairsFrenchAutre,
    # Requires manual download of data.
    # "storycloze_2016": storycloze.StoryCloze2016,
    # "storycloze_2018": storycloze.StoryCloze2018,
    # "sat": sat.SATAnalogies,
    **xcopa.construct_tasks(),
    **bigbench.create_all_tasks(),
    **xstorycloze.create_all_tasks(),
    **xwinograd.create_all_tasks(),
    **pawsx.construct_tasks(),
    **xnli.construct_tasks(),
    **mgsm.construct_tasks(),
}


ALL_TASKS = sorted(list(TASK_REGISTRY))

_EXAMPLE_JSON_PATH = "split:key:/absolute/path/to/data.json"


def add_json_task(task_name):
    """Add a JSON perplexity task if the given task name matches the
    JSON task specification.

    See `json.JsonPerplexity`.
    """
    if not task_name.startswith("json"):
        return

    def create_json_task():
        splits = task_name.split("=", 1)
        if len(splits) != 2 or not splits[1]:
            raise ValueError(
                "json tasks need a path argument pointing to the local "
                "dataset, specified like this: json="
                + _EXAMPLE_JSON_PATH
                + ' (if there are no splits, use "train")'
            )

        json_path = splits[1]
        if json_path == _EXAMPLE_JSON_PATH:
            raise ValueError(
                "please do not copy the example path directly, but substitute "
                "it with a path to your local dataset"
            )
        return lambda: json.JsonPerplexity(json_path)

    TASK_REGISTRY[task_name] = create_json_task()


def get_task(task_name):
    try:
        add_json_task(task_name)
        return TASK_REGISTRY[task_name]
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def get_task_dict(task_name_list: List[Union[str, lm_eval.base.Task]]):
    task_name_dict = {
        task_name: get_task(task_name)()
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list
        if not isinstance(task_object, str)
    }
    assert set(task_name_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {**task_name_dict, **task_name_from_object_dict}
