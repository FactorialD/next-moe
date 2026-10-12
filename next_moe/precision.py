"""FP16/BF16 model parameters with FP32 master weights for stable Adam updates."""
import torch


class MasterWeights:
    def __init__(self, parameters):
        self.model = list(parameters)
        self.master = [torch.nn.Parameter(p.detach().float().clone(), requires_grad=p.requires_grad)
                       for p in self.model]

    def copy_gradients(self):
        # GradScaler operates on FP32 masters, avoiding unscale_ on FP16 gradients.
        for model, master in zip(self.model, self.master):
            master.grad = None if model.grad is None else model.grad.detach().float()
            model.grad = None  # release low-precision gradients as we convert

    @torch.no_grad()
    def copy_to_model(self):
        for model, master in zip(self.model, self.master):
            model.copy_(master)

    def zero_grad(self):
        for model in self.model:
            model.grad = None
