from enum import StrEnum, auto, Enum

from algorithms.alg import *


class DatasetsEnum(StrEnum):
    amazon23office = auto()
    deezermarch = auto()


class AlgorithmsEnum(Enum):
    avgmatching = AverageQueryMatching
    crossmatching = CrossAttentionQueryMatching
    sparsematching = SparseMoEQueryMatching
    talkrec = TalkingToYourRecSys
    twotower = TwoTowerModel
