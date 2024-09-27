from collections import defaultdict
import math
from typing import List
import torch
from torch.optim.optimizer import Optimizer
import ipdb
from torch.optim import SparseAdam, Adam
from dataclasses import dataclass
from torch import Tensor
import random


class BlockLLM(Optimizer):
    def __init__(
        self,
        named_params,
        # named_parameter_list=[],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        sparsity_level=0.9,
        update_freq=1000,
        param_names=None,
        num_bottom_to_sample=1,
        # model=None,
    ):
        if not 0.0 <= sparsity_level < 1.0:
            raise ValueError(f"Invalid sparsity level: {sparsity_level}")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        # if not model:
        #     raise ValueError("Model is required")
        self.internal_optimizer = None  # will be set in reset_optimizer
        optim_groups = []

        # disable gradients for embedding and lm_head layers

        for param_name, param in named_params:
            # sparse_hook = self.sparse_update_hook(param_name)
            # for param in params:
            if ("embed" in param_name) or ("lm_head" in param_name):
                param.requires_grad = False
                continue
            if not param.requires_grad:
                continue
            state = {}
            state["name"] = param_name
            state["params"] = param
            optim_groups.append(state)

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            sparsity_level=sparsity_level,
            update_freq=update_freq,
            top_k_param_names=[],
            bottom_k_param_names=None,
            mask={},
            grad_norm_dict={},
            num_bottom_to_sample=num_bottom_to_sample,
        )
        super(BlockLLM, self).__init__(optim_groups, defaults)
        # super(BlockLLM, self).__init__(defaults)

        self.sparsity_level = sparsity_level
        self.update_freq = update_freq
        # self.param_names = param_names
        self.total_params = sum(
            p.numel() for group in self.param_groups for p in group["params"]
        )
        self.update_step = 0
        self.num_params_to_keep = int(
            (1 - self.sparsity_level) * len(self.param_groups)
        )

        # Compute top_k_params based on the weight magnitude
        self._choose_top_k_params_by_mag()

        # Make the requires_grad False for the remaining parameters
        for group in self.param_groups:
            name = group["name"]
            for param in group["params"]:
                if name not in self.defaults["top_k_param_names"]:
                    param.requires_grad = False
                    param.grad = None

        self._update_bottom_k_params()

    @torch.no_grad()
    def _choose_top_k_params_by_mag(self) -> List[str]:
        """
        Choose the top k parameters based on the weight magnitude.
        This function is intended to be used before the training starts.
        """
        mag_param_names_weighted = [
            (group["name"], param.data.abs().norm().item(), param)
            for group in self.param_groups
            for param in group["params"]
        ]
        # sorted_param_names = [
        #     (name, param)
        #     for name, _, param in sorted(
        #         mag_param_names_weighted,
        #         key=lambda x: x[1],
        #         reverse=True,
        #     )
        # ]
        random.shuffle(mag_param_names_weighted)
        sorted_param_names = [(x[0], x[2]) for x in mag_param_names_weighted]

        top_k_param_names = []
        top_k_params = []
        for name, param in sorted_param_names:
            if "embed" in name or "lm_head" in name:
                continue
            top_k_param_names.append(name)
            top_k_params.append(param)
            if len(top_k_params) >= self.num_params_to_keep:
                break
        # top_k_param_names = [
        #     name for name, _ in sorted_param_names[: self.num_params_to_keep]
        # ]
        # top_k_params = [
        #     param for _, param in sorted_param_names[: self.num_params_to_keep]
        # ]

        self.defaults["top_k_param_names"] = top_k_param_names
        self.reset_optimizer(top_k_params, mode="dense")
        del (
            sorted_param_names,
            mag_param_names_weighted,
        )
        torch.cuda.empty_cache()

    def _update_bottom_k_params(self):
        # Create a cycle for the bottom k parameters
        bottom_k_params = [
            param
            for group in self.param_groups
            for param in group["params"]
            if param not in self.defaults["top_k_param_names"]
        ]
        # self.defaults["bottom_k_param_names"] = itertools.cycle(bottom_k_params)
        self.defaults["bottom_k_param_names"] = bottom_k_params
        # del bottom_k_params

    def reset_optimizer(self, parameters, mode="sparse"):
        if mode == "sparse":
            self.internal_optimizer = SparseAdam(
                parameters, lr=self.defaults["lr"], betas=self.defaults["betas"]
            )
        else:
            self.internal_optimizer = Adam(
                parameters, lr=self.defaults["lr"], betas=self.defaults["betas"]
            )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if (self.update_step % self.update_freq == 0) and (self.update_step > 0):
            self._adjust_parameters()
            # self.update_step += 1
            # self._reset_state_dict()

        # if self.update_step == 1:
        #     self._adjust_parameters()

        self.update_step += 1
        self.internal_optimizer.step(closure)

        # Sample bottom k parameters and make them trainable

        # Update the gradient norms of all the trainable parameters
        # in the grad_norm_dict
        for group in self.param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                name = group["name"]
                self.defaults["grad_norm_dict"][name] = param.grad.abs().norm().item()

                if self.update_step % 10 == 0:
                    if name not in self.defaults["top_k_param_names"]:
                        param.requires_grad = False
                        param.grad = None
        torch.cuda.empty_cache()

        # Make the bottom k parameters not trainable

        if self.update_step % 10 == 0:
            for _ in range(self.defaults["num_bottom_to_sample"]):
                random.choice(self.defaults["bottom_k_param_names"]).requires_grad = (
                    True
                )
                # next(self.defaults["bottom_k_param_names"]).requires_grad = True
                # bottom_param = (self.defaults["bottom_k_param_names"])
                # bottom_param.requires_grad = True
        return loss

    def _adjust_parameters(self):
        """
        Find the top k parameters based on the gradient norms in the grad_norm_dict
        Update the bottom k parameters iterable
        """
        self.internal_optimizer = None

        # Find the top k parameters based on the gradient norms
        sorted_grad_norms = [
            name
            for name, _ in sorted(
                self.defaults["grad_norm_dict"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

        name_to_param = {group["name"]: group["params"] for group in self.param_groups}

        top_k_param_names = []
        top_k_params = []
        running_param_count = 0
        num_params_to_keep = int((1 - self.sparsity_level) * self.total_params)

        for name in sorted_grad_norms:
            if "embed" in name or "lm_head" in name:
                continue
            running_param_count += len(name_to_param[name])
            if running_param_count >= num_params_to_keep:
                break
            top_k_param_names.append(name)
            top_k_params.extend(name_to_param[name])

        # Update the top k parameters
        self.reset_optimizer(top_k_params, mode="dense")

        # Make the requires_grad False for the remaining parameters
        for group in self.param_groups:
            name = group["name"]
            for param in group["params"]:
                if name not in top_k_param_names:
                    param.requires_grad = False
                    param.grad = None

        # clear the gradient norms
        self.defaults["grad_norm_dict"] = {}

        # Update the bottom k parameters iterable
        self._update_bottom_k_params()
        del (
            top_k_params,
            sorted_grad_norms,
            top_k_param_names,
            name_to_param,
        )
        torch.cuda.empty_cache()
