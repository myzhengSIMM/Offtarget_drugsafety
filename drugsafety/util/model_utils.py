"""
Callback functions that can be invoked while fitting a KerasModel.
"""
import sys
import torch.nn as nn
import torch.nn.functional as F

from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.torch_model import TorchModel
import numpy as np
import torch
from torch.autograd import Variable

try:
    import torch.utils.tensorboard

    _has_tensorboard = True
except:
    _has_tensorboard = False
import time
import logging
import os

try:
    from collections.abc import Sequence as SequenceCollection
except:
    from collections import Sequence as SequenceCollection

from deepchem.data import Dataset, NumpyDataset
from deepchem.metrics import Metric
from deepchem.models import ValidationCallback
from deepchem.models.losses import Loss
from deepchem.models.models import Model
from deepchem.models.optimizers import Adam, Optimizer, LearningRateSchedule
from deepchem.trans import Transformer, undo_transforms
from deepchem.utils.evaluate import GeneratorEvaluator

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from deepchem.utils.typing import ArrayLike, LossFn, OneOrMany
from deepchem.models.wandblogger import WandbLogger

try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn(
            "W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable."
        )
    else:
        _has_wandb = True
except (ImportError, AttributeError):
    _has_wandb = False

logger = logging.getLogger(__name__)

# the optimizer defined
class deepchem_Adam():
    def __init__(self,
                 params,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-08,
                 weight_decay: float = 1e-08,
                 ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.params = params

    def _create_pytorch_optimizer(self, just_take_position):
        return torch.optim.Adam(self.params, self.learning_rate, (self.beta1, self.beta2), self.epsilon,
                                self.weight_decay)



class EarlystoppingCallback(ValidationCallback):
    def __init__(self,
                 dataset,
                 interval,
                 metrics,
                 save_dir = None,
                 save_metric = 0,
                 save_on_minimum = True,
                 patience = 1,
                 transformers = []):

        self.dataset = dataset
        self.interval = interval
        self.metrics = metrics
        self.save_dir = save_dir
        self.save_metric = save_metric
        self.save_on_minimum = save_on_minimum
        self._best_score = None
        self.transformers = transformers
        self.patience = patience 
        self.wait = patience 
        self.stop_training = False 

    def __call__(self, model, step):
        """Change the number of saved check point
    	"""

        if step % self.interval != 0:
            return
        scores = model.evaluate(self.dataset, self.metrics, self.transformers) 
        message = 'Step %d validation:' % step
        for key in scores:
            message += ' %s=%g' % (key, scores[key])
        # model.debug(message) 
        if model.tensorboard:
            for key in scores:
                model._log_scalar_to_tensorboard(key, scores[key],
                                                 model.get_global_step())
        if self.save_dir is not None:
            score = scores[self.metrics[self.save_metric].name] 
            if not self.save_on_minimum: 
                score = -score
            if self._best_score is None or score < self._best_score:
                model.save_checkpoint(max_checkpoints_to_keep=1, model_dir=self.save_dir)
                self._best_score = score
                self.patience = self.wait 
            else:
                self.patience -= 1 
        if model.wandb_logger is not None:
            # Log data to Wandb
            data = {'eval/' + k: v for k, v in scores.items()}
            model.wandb_logger.log_data(data, step, dataset_id=id(self.dataset))

        if self.patience == 0:
            self.stop_training = True