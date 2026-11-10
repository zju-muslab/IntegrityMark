import torch
import numpy as np
from itertools import chain


def my_step(opt, lr_sched, cur_iter, lr_steps=[2000,4000,8000,16000,32000]):
    opt.step()
    opt.zero_grad()
    lr_sched.step()


class ScheduledOptimMain:

    def __init__(self, encoder, decoder, train_config, model_config, current_step):
        self._optimizer = torch.optim.Adam(




            params=chain(decoder.parameters(), encoder.parameters()),
            betas=train_config["optimize"]["betas"],
            eps=train_config["optimize"]["eps"],
            weight_decay=train_config["optimize"]["weight_decay"],
        )
        self.n_up_steps = train_config["optimize"]["up_step"]
        self.anneal_steps = train_config["optimize"]["anneal_steps"]
        self.anneal_rate = train_config["optimize"]["anneal_rate"]
        self.init_lr = np.power(model_config["dim"]["embedding"], -0.5)
        self.current_step = current_step

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):

        self._optimizer.zero_grad()

    def load_state_dict(self, state_dict):
        state_dict['param_groups'] = self._optimizer.state_dict()['param_groups']
        self._optimizer.load_state_dict(state_dict)

    def _get_lr_scale(self):







        lr = np.power(self.n_up_steps, -1.5) * np.power(self.current_step, -0.2) * 100

        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr


class ScheduledOptimDisc:

    def __init__(self, model, train_config):

        self._optimizer = torch.optim.Adam(
            params=model.parameters(),
            betas=train_config["optimize"]["betas"],
            eps=train_config["optimize"]["eps"],
            weight_decay=train_config["optimize"]["weight_decay"],
        )
        self.init_lr = train_config["optimize"]["lr_disc"]
        self._init_learning_rate()

    def step_and_update_lr(self):
        self._optimizer.step()

    def zero_grad(self):

        self._optimizer.zero_grad()

    def load_state_dict(self, state_dict):
        state_dict['param_groups'] = self._optimizer.state_dict()['param_groups']
        self._optimizer.load_state_dict(state_dict)

    def _init_learning_rate(self):
        lr = self.init_lr
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
