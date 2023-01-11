from abc import ABC  # useless?
from typing import List, Tuple, Dict
import os
from logging import Logger
import yaml  # pip install pyaml
import json
import time
from datetime import timedelta
import copy

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn as nn
import logging

from sklearn.metrics import accuracy_score  # pip install scikit-learn
from numpy import array


logger = logging.getLogger()


class BaseModel(ABC):
    def __init__(self, logger: Logger = None, print_progress: bool = True,
                 device: str = 'cuda:0', **kwargs):
        """

        """
        # where to print info
        self.print_fn = logger.info if logger else print

        self.device = device
        self.print_progress = print_progress

        self.outputs = {}
        # placeholder for any output to be saved in YAML
        self.extra_checkpoint_keys = []
        # list of attribute name to save in checkpoint

    def save_outputs(self, export_path: str):
        """
        Save the output attribute dictionnary as a YAML or JSON
        specified by export_path.
        """
        if os.path.splitext(export_path)[-1] in ['.yml', '.yaml']:
            with open(export_path, "w") as f:
                yaml.dump(self.outputs, f)
        elif os.path.splitext(export_path)[-1] == '.json':
            with open(export_path, "w") as f:
                json.dump(self.outputs, f)

    @staticmethod
    def print_progessbar(n: int, max: int, name: str = '', size: int = 10,
                         end_char: str = '', erase: bool = False):
        """
        Print a progress bar. To be used in a for-loop and called at each
        iteration with the iteration number and the max number of iteration.
        ------------
        INPUT
            |---- n (int) the iteration current number
            |---- max (int) the total number of iteration
            |---- name (str) an optional name for the progress bar
            |---- size (int) the size of the progress bar
            |---- end_char (str) the print end parameter to used in the end of
            |                    the progress bar (default is '')
            |---- erase (bool) to erase the progress bar when 100% is reached.
        OUTPUT
            |---- None
        """
        frmt = f"0{len(str(max))}d"
        print(f'{name} {n+1:{frmt}}/{max:{frmt}}'.ljust(len(name) + 12)
              + f'|{"█"*int(size*(n+1)/max)}'.ljust(size+1) +
              f'| {(n+1)/max:.1%}'.ljust(6),
              end='\r')

        if n+1 == max:
            if erase:
                print(' '.ljust(len(name) + size + 40), end='\r')
            else:
                print('')


def set_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(
            f"CUDA device name is: {torch.cuda.get_device_name(device)}"
        )
    else:
        logger.warning(f"Running on {device}, not on CUDA")
    return device


class BaseModelSingle(BaseModel):
    def __init__(self, net: nn.Module, opt: Optimizer = None,
                 sched: _LRScheduler = None, logger: Logger = None,
                 print_progress: bool = True,
                 device: str = 'cuda:0', **kwargs):
        """
        Abstract class defining a moodel based on Pytorch.
        It allows to save/load the model and train/evaluate it.
        Classes inheriting from the BaseModel needs to be
         initialized with a nn.Modules.
        This network can be trained using the passed optimizer/lr_scheduler
         with the self.train() methods.
        To be used, the children class must define two
        abstract methods:
            1. `forward_loss(data: Tuple[Tensor])`:
                define the processing of 1 batch provided by the DataLoader.
                `data` is the tuple of tensors given by the DataLoader.
                This method should thus define how the data is
                    i) unpacked
                    ii) how the forward pass with self.net is done
                    iii) and how the loss is computed. The method should then
               return the loss.
            2. `validate(loader: DataLoader)`:
                define how the model is validated at each epoch.
                It takes a DataLoader for the validation data as input and
                 should return a dictionnary of properties to print in the
                 epoch summary (as {property_name : str_property_value}).
                 No validation is performed if no valid_loader is passed
                  to self.train()

        Note: the BaseModel has a dictionnary as attributes (self.outputs)
               that allow to store some values (training time,
               validation scores, epoch evolution, etc).
                This dictionnary can be saved as a YAML file using the
                 save_outputs method.
                 Any other values can be added to the
                  self.outputs using self.outputs["key"] = value.

              If Logger is None, the outputs are displayed using `print`.
        """
        super().__init__(
            logger=logger,
            print_progress=print_progress,
            device=device,
            **kwargs
        )

        self.net = net
        self.net = self.net.to(device)
        self.best_net = net
        self.best_metric = None
        self.optimizer = opt
        self.lr_scheduler = sched
        self.logger = logger

    def train(self, n_epochs: int, train_loader: DataLoader,
              valid_loader: DataLoader = None, extra_valid_args: List = [],
              extra_valid_kwargs: Dict = dict(), checkpoint_path: str = None,
              checkpoint_freq: int = 10, save_best_key: str = None,
              minimize_metric: bool = True, min_epoch_best: int = 0):
        """
        Train the self.net using the optimizer and scheduler using the data
         provided by the train_loader. At each epoch, the model can be
         validated using the valid_loader (if a valid loader is provided,
         the method self.validate must
          be implemented in the children). The model and training state is
           loaded/saved in a .pt file if checkpoint_path
            is provided. The model is then saved every checkpoint_freq epoch.

        The best model can be saved over the training processed based on one
         of the validation metric provided by the self.validate output
         dictionnary. The metric to use is specified by the string
         `save_best_key` and the argument `minimize_metric` define whether
         the metric must be minimized or maximized. A mininumm number of epoch
         to be performed before selcting the best model can be specified with
         'min_epoch_best'.
        """
        if self.optimizer is None:
            raise ValueError(
                "An optimizer must be provided to train the model."
            )

        # Load checkpoint if any
        if checkpoint_path:
            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location=self.device)
                n_epoch_finished = checkpoint['n_epoch_finished']
                self.net.load_state_dict(checkpoint['net_state'])
                self.net = self.net.to(self.device)
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                if save_best_key:
                    best_metric = checkpoint['best_metric']
                    best_epoch = checkpoint['best_epoch']
                    self.best_net.load_state_dict(checkpoint['best_net_state'])
                    self.best_net = self.best_net.to(self.device)

                if self.lr_scheduler:
                    self.lr_scheduler.load_state_dict(checkpoint['lr_state'])

                epoch_loss_list = checkpoint['loss_evolution']

                for k in self.extra_checkpoint_keys:
                    setattr(self, k, checkpoint[k])

                self.print_fn(
                    f'Resuming from Checkpoint with {n_epoch_finished} '
                    f'epoch finished.')
            except FileNotFoundError:
                self.print_fn('No Checkpoint found. Training from beginning.')
                n_epoch_finished = 0
                epoch_loss_list = []  # Placeholder for epoch evolution
        else:
            self.print_fn('No Checkpoint used. Training from beginning.')
            n_epoch_finished = 0
            epoch_loss_list = []

        self.net = self.net.to(self.device)

        # Train Loop
        for epoch in range(n_epoch_finished, n_epochs):
            self.net.train()
            epoch_start_time = time.time()

            # Train Loop
            for b, data in enumerate(train_loader):
                # Gradient descent step
                self.optimizer.zero_grad()
                loss = self.forward_loss(data)
                # recover returned loss(es) value(s)
                if isinstance(loss, tuple):
                    loss, all_losses = loss
                    if b == 0:
                        train_outputs = {
                            name: 0.0 for name in all_losses.keys()}
                    train_outputs = {
                        name: (value + all_losses[name].item()
                               if isinstance(all_losses[name], torch.Tensor)
                               else value + all_losses[name])
                        for name, value in train_outputs.items()
                    }
                else:
                    if b == 0:
                        train_outputs = {'Loss': 0.0}
                    train_outputs["Loss"] += loss.item()

                loss.backward()
                self.optimizer.step()

                if self.print_progress:
                    self.print_progessbar(b, train_loader.__len__(
                    ), name='Train Batch', size=100, erase=True)

            # Validate Loop
            if valid_loader:
                self.net.eval()
                with torch.no_grad():
                    for b, data in enumerate(valid_loader):
                        loss = self.validate(
                            data, *extra_valid_args, **extra_valid_kwargs)
                        if isinstance(loss, tuple):
                            loss, all_losses = loss
                            if b == 0:
                                valid_outputs = {
                                    name: 0.0 for name in all_losses.keys()}
                            valid_outputs = {
                                name: (value + all_losses[name].item()
                                       if isinstance(
                                    all_losses[name], torch.Tensor
                                )
                                    else value + all_losses[name])
                                for name, value in valid_outputs.items()
                            }
                        else:
                            if b == 0:
                                valid_outputs = {'Valid Loss': 0.0}
                            valid_outputs["Valid Loss"] += loss.item()

                        if self.print_progress:
                            self.print_progessbar(b, valid_loader.__len__(
                            ), name='Valid Batch', size=100, erase=True)

            else:
                valid_outputs = {}

            # print epoch stat
            frmt = f"0{len(str(n_epochs))}"
            self.print_fn(
                f"Epoch {epoch+1:{frmt}}/{n_epochs:{frmt}} | "  # Epoch number
                # Time
                f"Time {timedelta(seconds=time.time()-epoch_start_time)} | "
                + "".join([f"{name} {loss_i / train_loader.__len__():.5f} | "
                           for name, loss_i in train_outputs.items()])  # Train
                + "".join([f"{name} {loss_i / valid_loader.__len__():.5f} | "
                           for name, loss_i in valid_outputs.items()]))  # Val

            epoch_loss_list.append([epoch+1,
                                    {name: loss/train_loader.__len__()
                                     for name, loss in train_outputs.items()},
                                    {name: loss/valid_loader.__len__()
                                     for name, loss in valid_outputs.items()}])

            # Scheduler steps
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # Update best model
            if save_best_key:
                assert save_best_key in valid_outputs.keys(
                ), f"`save_best_key` must be present in the validation output"
                f"dict to save the best model."
                # initialize if first epoch
                if epoch == 0:
                    best_metric = valid_outputs[save_best_key]
                    best_epoch = epoch+1
                    self.best_net = copy.deepcopy(self.net)
                # update best net
                if (
                    minimize_metric and
                    valid_outputs[save_best_key] < best_metric
                ) or (
                        not minimize_metric and
                        valid_outputs[save_best_key] > best_metric
                ) or epoch < min_epoch_best:
                    best_metric = valid_outputs[save_best_key]
                    best_epoch = epoch+1
                    self.best_net = copy.deepcopy(self.net)

            # Save checkpoint
            if (epoch+1) % checkpoint_freq == 0 and checkpoint_path:
                checkpoint = {
                    'n_epoch_finished': epoch+1,
                    'net_state': self.net.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'loss_evolution': epoch_loss_list
                }
                if save_best_key:
                    checkpoint['best_metric'] = best_metric
                    checkpoint['best_epoch'] = best_epoch
                    checkpoint['best_net_state'] = self.best_net.state_dict()
                if self.lr_scheduler:
                    checkpoint['lr_state'] = self.lr_scheduler.state_dict()

                for k in self.extra_checkpoint_keys:
                    checkpoint[k] = getattr(self, k)

                torch.save(checkpoint, checkpoint_path)
                self.print_fn('\tCheckpoint saved.')

        self.outputs['train_evolution'] = epoch_loss_list
        if save_best_key:
            self.outputs['best_model'] = {
                save_best_key: best_metric, 'epoch': best_epoch}

    def save(self, export_path: str):
        """
        Save model state dictionnary at the export_path.
        """
        torch.save(self.net.state_dict(), export_path)

    def load(self, import_path: str, map_location: str = 'cuda:0'):
        """
        Load the model state dictionnary at the import path on the device
        specified by map_location.
        """
        device = set_device()
        loaded_state_dict = torch.load(
            import_path, map_location=device)  # , on=device)
        self.net.load_state_dict(loaded_state_dict)


class Classifier(BaseModelSingle):
    """ """

    def __init__(
        self, net: nn.Module, opt: Optimizer = None,
        sched: _LRScheduler = None, logger: Logger = None,
        print_progress: bool = True, device: str = 'cuda:0', **kwargs
    ):
        super().__init__(net, opt=opt, sched=sched, logger=logger,
                         print_progress=print_progress,
                         device=device, **kwargs)

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def forward_loss(self, data: Tuple[Tensor]) -> Tensor:
        """  """
        input, label = data
        input = input.to(self.device)
        label = label.to(self.device).long()

        output = self.net(input)
        loss = self.loss_fn(output, label)

        pred = torch.argmax(output, dim=1)
        pred_label = list(zip(pred.cpu().data.tolist(),
                          label.cpu().data.tolist()))

        pred, label = zip(*pred_label)
        acc = accuracy_score(array(label), array(pred))

        return loss, {"Loss": loss, "Train Accuracy": acc}

    def predict(self, loader):
        """  """
        self.net.eval()
        labels = []
        preds = []
        len_loader = len(loader)
        losses = list(range(len_loader))
        with torch.no_grad():
            for b, data in enumerate(loader):
                self.print_progessbar(b, len_loader, name='Prediction')
                input, label = data
                input = input.to(self.device)
                label = label.to(self.device).long()

                output = self.net(input)
                pred = torch.argmax(output, dim=1)

                preds += pred.cpu().data.tolist()
                labels += label.cpu().data.tolist()
                losses[b] = self.loss_fn(output, label).item()
            print()

        return preds, labels, losses

    def validate(self, data):
        """  """
        input, label = data
        input = input.to(self.device)
        label = label.to(self.device).long()

        output = self.net(input)
        loss = self.loss_fn(output, label).item()

        pred = torch.argmax(output, dim=1)
        pred_label = list(zip(pred.cpu().data.tolist(),
                          label.cpu().data.tolist()))

        pred, label = zip(*pred_label)
        acc = accuracy_score(array(label), array(pred))

        return loss, {"Valid Loss": loss, "Valid Accuracy": acc}
