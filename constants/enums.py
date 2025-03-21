from enum import StrEnum, auto, Enum

from algorithms.alg import *


class DatasetsEnum(StrEnum):
    amazon23office = auto()


class AlgorithmsEnum(Enum):
    avgmatching = AverageQueryMatching
    crossmatching = CrossAttentionQueryMatching