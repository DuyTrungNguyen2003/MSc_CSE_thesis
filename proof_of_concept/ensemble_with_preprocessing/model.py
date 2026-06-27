import torch
import torch.nn as nn

from L0_layer import L0Dense

class L0LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size, beta_ema, L0_LAMBDA, scaled_weight_decay, droprate_init, temperature, local_rep):
        super().__init__()
        self.beta_ema = beta_ema
        self.output_layer = L0Dense(
            in_features=input_size,
            out_features=output_size,
            lamba=L0_LAMBDA,
            weight_decay=scaled_weight_decay,
            droprate_init=droprate_init,
            temperature=temperature,
            local_rep=local_rep
        )

        # EMA: shadow copies of all parameters, kept on the same device as the model.
        # Initialised to the current parameter values (step 0).
        # We store them as plain tensors (not Parameters) so the optimiser ignores them.
        if self.beta_ema > 0.:
            print(f'Using temporal averaging with beta: {self.beta_ema}')
            self.avg_param = [p.data.clone() for p in self.parameters()]
            self.steps_ema = 0.

    def forward(self, x):
        return self.output_layer(x)

    def regularization(self):
        return self.output_layer.regularization()

    # ---- EMA helpers (mirroring models.py) ----

    def update_ema(self):
        """Call once per training step, after optimizer.step()."""
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        """Swap live weights for bias-corrected EMA weights (use before validation/test)."""
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema ** self.steps_ema))

    def load_params(self, params):
        """Restore a previously saved list of parameter tensors."""
        for p, saved in zip(self.parameters(), params):
            p.data.copy_(saved)

    def get_params(self):
        """Return a snapshot of the current live parameter tensors."""
        return [p.data.clone() for p in self.parameters()]