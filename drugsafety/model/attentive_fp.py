import sys
import torch.nn as nn
import torch.nn.functional as F

from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.torch_model import TorchModel
import numpy as np
import torch

import time
import logging
import os
try:
    from collections.abc import Sequence as SequenceCollection
except:
    from collections import Sequence as SequenceCollection

from dgllife.model import AttentiveFPPredictor

from deepchem.data import Dataset, NumpyDataset
from deepchem.metrics import Metric
from deepchem.models.losses import Loss
from deepchem.models.models import Model
from deepchem.models.optimizers import Adam, Optimizer, LearningRateSchedule
from deepchem.trans import Transformer, undo_transforms
from deepchem.utils.evaluate import GeneratorEvaluator
from deepchem.models.torch_models import AttentiveFP
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from deepchem.utils.typing import ArrayLike, LossFn, OneOrMany

from util.model_utils import deepchem_Adam



class AttentiveFPModel(TorchModel):

    def __init__(self,
                             n_tasks: int,
                             num_layers: int = 2,
                             num_timesteps: int = 2,
                             graph_feat_size: int = 200,
                             dropout: float = 0.,
                             mode: str = 'regression',
                             number_atom_features: int = 30,
                             number_bond_features: int = 11,
                             n_classes: int = 2,
                             self_loop: bool = True,
                             learning_rate = 1e-4,
                             weight_decay = 1e-7,
                             device: Optional[torch.device] = None,
                             logger = None,
                             **kwargs):

        if logger is not None:
            self.debug, self.info = logger.debug, logger.info
        else:
            self.debug = self.info = print

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model = AttentiveFP(
                n_tasks=n_tasks,
                num_layers=num_layers,
                num_timesteps=num_timesteps,
                graph_feat_size=graph_feat_size,
                dropout=dropout,
                mode=mode,
                number_atom_features=number_atom_features,
                number_bond_features=number_bond_features,
                n_classes=n_classes)
        if mode == 'regression':
            loss: Loss = L2Loss()
            output_types = ['prediction']
        else:
            loss = SparseSoftmaxCrossEntropy()
            output_types = ['prediction', 'loss'] 

        optimizer = self.build_optimizer()
        super(AttentiveFPModel, self).__init__(
                self.model, loss=loss, output_types=output_types, learning_rate=self.learning_rate, optimizer=optimizer, device=device, **kwargs)

        self._self_loop = self_loop

    def fit(self,
            dataset: Dataset,
            nb_epoch: int = 10,
            max_checkpoints_to_keep: int = 5,
            checkpoint_interval: int = 1000,
            deterministic: bool = False,
            restore: bool = False,
            variables: Optional[List[torch.nn.Parameter]] = None,
            loss: Optional[LossFn] = None,
            callbacks: Union[Callable, List[Callable]] = [],
            all_losses: Optional[List[float]] = None) -> float:

        self.num_training_data = dataset.get_shape()[0][0]

        return self.fit_generator(
            self.default_generator(
                dataset, epochs=nb_epoch,
                deterministic=deterministic), max_checkpoints_to_keep,
            checkpoint_interval, restore, variables, loss, callbacks, all_losses)

    def fit_generator(self,
                      generator: Iterable[Tuple[Any, Any, Any]],
                      max_checkpoints_to_keep: int = 5,
                      checkpoint_interval: int = 1000,
                      restore: bool = False,
                      variables: Optional[List[torch.nn.Parameter]] = None,
                      loss: Optional[LossFn] = None,
                      callbacks: Union[Callable, List[Callable]] = [],
                      all_losses: Optional[List[float]] = None) -> float:

        if not isinstance(callbacks, SequenceCollection):
            callbacks = [callbacks]
        self._ensure_built()
        self.model.train()
        avg_loss = 0.0
        last_avg_loss = 0.0
        averaged_batches = 0
        if loss is None:
            loss = self._loss_fn
        if variables is None:
            optimizer = self._pytorch_optimizer
            lr_schedule = self._lr_schedule
        else:
            var_key = tuple(variables)
            if var_key in self._optimizer_for_vars:
                optimizer, lr_schedule = self._optimizer_for_vars[var_key]
            else:
                optimizer = self.optimizer._create_pytorch_optimizer(variables)
                if isinstance(self.optimizer.learning_rate, LearningRateSchedule):
                    lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(
                        optimizer)
                else:
                    lr_schedule = None
                self._optimizer_for_vars[var_key] = (optimizer, lr_schedule)
        time1 = time.time()

        # Main training loop.

        for batch in generator:
            if restore:
                self.restore()
                restore = False
            inputs: OneOrMany[torch.Tensor]
            inputs, labels, weights = self._prepare_batch(batch) 

            # Execute the loss function, accumulating the gradients.

            if isinstance(inputs, list) and len(inputs) == 1:
                inputs = inputs[0]

            optimizer.zero_grad()
            outputs = self.model(inputs) 
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
            if self._loss_outputs is not None:
                outputs = [outputs[i] for i in self._loss_outputs]
            batch_loss = loss(outputs, labels, weights) 
            batch_loss.backward()
            optimizer.step()
            if lr_schedule is not None:
                lr_schedule.step()
            self._global_step += 1
            current_step = self._global_step
            current_epoch = int(current_step * self.batch_size / self.num_training_data)
            avg_loss += batch_loss

            # Report progress and write checkpoints.
            averaged_batches += 1
            should_log = (current_step % self.log_frequency == 0)
            if should_log:
                avg_loss = float(avg_loss) / averaged_batches
                self.debug(
                    'Ending global_step %d, epoch %d: Average loss %g' % (current_step, current_epoch, avg_loss))
                if all_losses is not None:
                    all_losses.append(avg_loss)
                # Capture the last avg_loss in case of return since we're resetting to 0 now
                last_avg_loss = avg_loss
                avg_loss = 0.0
                averaged_batches = 0
            if self.tensorboard and should_log:
                self._log_scalar_to_tensorboard('loss', batch_loss, current_step)
            if (self.wandb_logger is not None) and should_log:
                all_data = dict({'train/loss': batch_loss})
                self.wandb_logger.log_data(all_data, step=current_step)
            # if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval - 1:
            #     self.save_checkpoint(max_checkpoints_to_keep)
            for c in callbacks:
                c(self, current_step) 
            if any([c.stop_training for c in callbacks]):  # for early stopping callback
                self.debug(f"early stopping at step {current_step}, epoch {current_epoch}")
                break

        # Report final results.
        if averaged_batches > 0:
            avg_loss = float(avg_loss) / averaged_batches
            self.debug(
                'Ending global_step %d, epoch %d: Average loss %g' % (current_step, current_epoch, avg_loss))
            if all_losses is not None:
                all_losses.append(avg_loss)
            last_avg_loss = avg_loss

        # if checkpoint_interval > 0:
        #     self.save_checkpoint(max_checkpoints_to_keep)

        time2 = time.time()
        self.debug("TIMING: model fitting took %0.3f s" % (time2 - time1))
        return last_avg_loss
    
    

    def _prepare_batch(self, batch):
        """Create batch data for AttentiveFP.

        Parameters
        ----------
        batch: tuple
            The tuple is ``(inputs, labels, weights)``.

        Returns
        -------
        inputs: DGLGraph
            DGLGraph for a batch of graphs.
        labels: list of torch.Tensor or None
            The graph labels.
        weights: list of torch.Tensor or None
            The weights for each sample or sample/task pair converted to torch.Tensor.
        """
        try:
            import dgl
        except:
            raise ImportError('This class requires dgl.')

        inputs, labels, weights = batch
        dgl_graphs = [
                graph.to_dgl_graph(self_loop=self._self_loop) for graph in inputs[0]
        ]
        inputs = dgl.batch(dgl_graphs).to(self.device)
        _, labels, weights = super(AttentiveFPModel, self)._prepare_batch(
                ([], labels, weights))
        return inputs, labels, weights
    
    @staticmethod
    def apply_dropout(m):
        if type(m) == torch.nn.Dropout:
            m.train()
    
    def _predict(self, generator: Iterable[Tuple[Any, Any, Any]],
                             transformers: List[Transformer], uncertainty,
                             other_output_types: Optional[OneOrMany[str]]):
        results: Optional[List[List[np.ndarray]]] = None
        variances: Optional[List[List[np.ndarray]]] = None
        if uncertainty and (other_output_types is not None):
            raise ValueError(
                'This model cannot compute uncertainties and other output types simultaneously. Please invoke one at a time.'
            )
        
        if uncertainty:
            if self._variance_outputs is None or len(self._variance_outputs) == 0:
                raise ValueError('This model cannot compute uncertainties')
            if len(self._variance_outputs) != len(self._prediction_outputs):
                raise ValueError(
                    'The number of variances must exactly match the number of outputs')
        if other_output_types:
            if self._other_outputs is None or len(self._other_outputs) == 0:
                raise ValueError(
                    'This model cannot compute other outputs since no other output_types were specified.'
                )
        self._ensure_built()
        self.model.eval()
        
        for batch in generator:
            inputs, labels, weights = batch
            inputs, _, _ = self._prepare_batch((inputs, None, None))

            # Invoke the model.
            if isinstance(inputs, list) and len(inputs) == 1:
                inputs = inputs[0]
            output_values = self.model(inputs)
            if isinstance(output_values, torch.Tensor):
                output_values = [output_values]
            output_values = [t.detach().cpu().numpy() for t in output_values]
            # Apply tranformers and record results.
            if uncertainty:
                var = [output_values[i] for i in self._variance_outputs]
                if variances is None:
                    variances = [var]
                else:
                    for i, t in enumerate(var):
                        variances[i].append(t)
            access_values = []
            if other_output_types:
                access_values += self._other_outputs
            elif self._prediction_outputs is not None:
                access_values += self._prediction_outputs

            if len(access_values) > 0:
                output_values = [output_values[i] for i in access_values]

            if len(transformers) > 0:
                if len(output_values) > 1:
                    raise ValueError(
                        "predict() does not support Transformers for models with multiple outputs."
                    )
                elif len(output_values) == 1:
                    output_values = [undo_transforms(output_values[0], transformers)]
            if results is None:
                results = [[] for i in range(len(output_values))]
            for i, t in enumerate(output_values):
                results[i].append(t)

        # Concatenate arrays to create the final results.
        final_results = []
        final_variances = []
        if results is not None:
            for r in results:
                final_results.append(np.concatenate(r, axis=0))
        if uncertainty and variances is not None:
            for v in variances:
                final_variances.append(np.concatenate(v, axis=0))
            return zip(final_results, final_variances)
           
        if len(final_results) == 1:
            return final_results[0]
        else:
            return final_results


    def restore(self,
                checkpoint: Optional[str] = None,
                model_dir: Optional[str] = None) -> None:
        """Reload the values of all variables from a checkpoint file.

        Parameters
        ----------
        checkpoint: str
          the path to the checkpoint file to load.  If this is None, the most recent
          checkpoint will be chosen automatically.  Call get_checkpoints() to get a
          list of all available checkpoints.
        model_dir: str, default None
          Directory to restore checkpoint from. If None, use self.model_dir.  If
          checkpoint is not None, this is ignored.
        """
        # self._ensure_built()
        if checkpoint is None:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            if len(checkpoints) == 0:
                raise ValueError('No checkpoint found')
            checkpoint = checkpoints[0]
        #state = torch.load(checkpoint)
        state = torch.load(checkpoint, map_location=self.device) 
        loaded_state_dict = state['model_state_dict'] 

        model_state_dict = self.model.state_dict() 
        pretrained_state_dict = {}
        for param_name in loaded_state_dict.keys():
            new_param_name = param_name
            if new_param_name not in model_state_dict:
                self.debug(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
            elif model_state_dict[new_param_name].shape != loaded_state_dict[param_name].shape:
                self.debug(f'Pretrained parameter "{param_name}" '
                      f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                      f'model parameter of shape {model_state_dict[new_param_name].shape}.')
            else:
                # print(f'Loading pretrained parameter "{param_name}".')
                pretrained_state_dict[new_param_name] = loaded_state_dict[param_name] 
        model_state_dict.update(pretrained_state_dict) 
        self.model.load_state_dict(model_state_dict)
        # self._pytorch_optimizer.load_state_dict(state['optimizer_state_dict'])
        self._pytorch_optimizer = self.build_optimizer()._create_pytorch_optimizer(None)
        self._global_step = state['global_step']
    

    def build_optimizer(self):

        return deepchem_Adam(self.model.parameters(), learning_rate=self.learning_rate, weight_decay=self.weight_decay) 