

from .mnli import (
    NLIRelationClassifierWithMappingHead,
    NLIRelationClassifier,
    REInputFeatures,
)
from .tacred import TACRED_LABELS, TACREDClassifier

__all__ = [
    "REInputFeatures",
    "NLIRelationClassifier",
    "NLIRelationClassifierWithMappingHead",
    "TACREDClassifier",
    "TACRED_LABELS",
]

# import sys
# import os
# from pathlib import Path
# CURR_FILE_PATH = (os.path.abspath(__file__))
# PATH = Path(CURR_FILE_PATH)
# CURR_DIR = str(PATH.parent.absolute())
# sys.path.append(str(PATH.parent.parent.parent.absolute()))
# sys.path.append(str(PATH.parent.parent.parent.parent.absolute()))
# sys.path.append(CURR_DIR)

# from URE_mnli.relation_classification.mnli import (
#     NLIRelationClassifierWithMappingHead,
#     NLIRelationClassifier,
#     REInputFeatures,
# )
# from URE_mnli.relation_classification.tacred import TACRED_LABELS, TACREDClassifier

# __all__ = [
#     "REInputFeatures",
#     "NLIRelationClassifier",
#     "NLIRelationClassifierWithMappingHead",
#     "TACREDClassifier",
#     "TACRED_LABELS",
# ]