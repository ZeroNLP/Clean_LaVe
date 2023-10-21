import sys
import os
from pathlib import Path
CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(str(PATH.parent.parent.parent.absolute()))
sys.path.append(str(PATH.parent.parent.parent.parent.absolute()))
sys.path.append(CURR_DIR)
import arguments
import gc
from typing import List

import numpy as np
import torch

try:
    import transformers

    transformers.logging.set_verbosity_error()
except ImportError:
    pass


def np_softmax(x, dim=-1):
    e = np.exp(x)
    return e / np.sum(e, axis=dim, keepdims=True)


def np_sigmoid(x, dim=-1):
    return 1 / (1 + np.exp(-x))


class Classifier(object):
    def __init__(
        self,
        labels: List[str],
        pretrained_model: str = arguments.model_path,
        use_cuda=True,
        half=False,
        verbose=True,
    ):
        super().__init__()

        self.device = torch.device("cuda:{}".format(arguments.cuda_index))
        print("use cuda device:",self.device)
        self.labels = labels
        self.use_cuda = use_cuda
        self.half = half
        self.verbose = verbose

        # Supress stdout printing for model downloads
        if not verbose:
            sys.stdout = open(os.devnull, "w")
            self._initialize(pretrained_model)
            sys.stdout = sys.__stdout__
        else:
            self._initialize(pretrained_model)

        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        if self.use_cuda and self.half and torch.cuda.is_available():
            self.model = self.model.half()

    def _initialize(self, pretrained_model):
        raise NotImplementedError

    def __call__(self, context, batch_size=1):
        raise NotImplementedError

    def clear_gpu_memory(self):
        self.model.cpu()
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
