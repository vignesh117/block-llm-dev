import math
import torch
from torch.optim.optimizer import Optimizer


class BlockLLM(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        sparsity_level=0.9,
        update_freq=1000,
        param_names=None,
        model=None,
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
        if not model:
            raise ValueError("Model is required")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            sparsity_level=sparsity_level,
            update_freq=update_freq,
            param_names=param_names,
            model=model,
            top_k_params=[],
            num_params_to_keep=0,  # this is dynamic, selected after layers are selected
        )
        super(BlockLLM, self).__init__(params, defaults)

        self.sparsity_level = sparsity_level
        self.update_freq = update_freq
        self.param_names = param_names
        self.param_visit_count = (
            {name: 0 for name in param_names} if param_names else {}
        )
        self.total_params = sum(
            p.numel() for group in self.param_groups for p in group["params"]
        )
        self.param_dist_weights = self._compute_param_dist_weights()
        self.update_step = 0

    def _compute_param_dist_weights(self):
        weights = {}
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group["params"]):
                name = (
                    self.param_names[i * len(group["params"]) + j]
                    if self.param_names
                    else f"param_{i}_{j}"
                )
                weights[name] = (i * len(group["params"]) + j + 1) / len(
                    self.param_groups
                )
        return weights

    def _bookkeep_mask(self, curr_mask: torch.Tensor, p_name):
        # if p_name not in self.updated_parameters:
        if self.updated_parameters.get(p_name) is None:
            self.updated_parameters[p_name] = curr_mask

        else:
            self.updated_parameters[p_name] = (
                curr_mask.bool() | self.updated_parameters[p_name]
            )

    def _compute_mask(self, pname, grad, quantile, k):
        """Compute the mask by selecting the top k gradients."""
        if pname in self.defaults["mask"]:
            return self.defaults["mask"][pname]
        else:
            # Flatten the gradient tensor and get the absolute values
            k = min(k, len(grad.view(-1)))
            flat_grad = torch.abs(grad).flatten()
            _, top_k_indices = torch.topk(flat_grad, k)
            # Initialize a mask of zeros
            mask = torch.zeros_like(flat_grad, dtype=torch.bool)

            # Set the top k positions to 1
            mask[top_k_indices] = 1
            # Reshape the mask to the original gradient shape and convert to sparse
            mask = mask.view_as(grad)
            self.defaults["mask"][pname] = mask
            return mask

    # def _compute_mask(self, grad, k):
    #     k = min(k, grad.numel())
    #     flat_grad = torch.abs(grad).flatten()
    #     _, top_k_indices = torch.topk(flat_grad, k)
    #     mask = torch.zeros_like(flat_grad, dtype=torch.bool)
    #     mask[top_k_indices] = 1
    #     return mask.view_as(grad)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if self.update_step % self.update_freq == 0:
            self._adjust_parameters()

        self.update_step += 1
        k_remaining = self.defaults["num_params_to_keep"]

        for group in self.param_groups:
            for pname, p in self.defaults["model"].named_parameters():
                if p.grad is None:
                    continue
                if pname not in self.defaults.get("top_k_params", []):
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = (
                    group["lr"]
                    * math.sqrt(1 - beta2 ** state["step"])
                    / (1 - beta1 ** state["step"])
                )
                k_remaining -= len((exp_avg / denom).view(-1))

                # if self.update_step % group["update_freq"] == 0:
                #     # Compute mask
                #     mask = self._compute_mask(pname, exp_avg / denom, k_remaining)
                #     k_remaining -= mask.sum().item()
                #     state["mask"] = mask
                # else:
                #     mask = state.get("mask", torch.ones_like(p.data, dtype=torch.bool))

                # self._bookkeep_mask(mask, pname)

                mask = state.get("mask", torch.ones_like(p.data, dtype=torch.bool))

                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                p.data.masked_fill_(~mask, 0)

        return loss

    def _adjust_parameters(self):
        """
        Adjust parameters based on gradients magnitude and visit count.
        This function is called every update_freq steps.
        It selects the top k parameters based on gradients magnitude and visit count.
        It then updates the visit count and the defaults with the top k parameters.
        """
        print("Adjusting parameters")
        grad_mag_param_names_weighted = [
            (name, param.grad.abs().norm().item())
            for name, param in self.defaults["model"].named_parameters()
            if param.grad is not None
        ]

        # Sort parameter names based on gradients magnitude and visit count
        sorted_param_names = [
            name
            for name, _ in sorted(
                grad_mag_param_names_weighted,
                key=lambda x: x[1] / (self.param_visit_count.get(x[0], 1) + 1e-8),
                reverse=True,
            )
        ]
        num_params_to_keep = int((1 - self.sparsity_level) * self.total_params)

        top_k_params = []
        running_param_count = 0
        for name in sorted_param_names:
            param = self._get_parameter_by_name(self.defaults["model"], name)
            if running_param_count < num_params_to_keep:
                top_k_params.append(name)
                running_param_count += param.numel()
            else:
                break

        # Update visit count
        for name in top_k_params:
            self.param_visit_count[name] = self.param_visit_count.get(name, 0) + 1

        # Update defaults with top_k_params
        self.defaults["top_k_params"] = top_k_params

    def _get_parameter_by_name(self, model, name):
        for param_name, param in model.named_parameters():
            if param_name == name:
                return param
        return None
