from contextlib import contextmanager

import torch


@contextmanager
def ema_scope(use_ema, model_ema, models):
    """Used to validate the model with EMA parameters without affecting optimization."""
    if use_ema:
        model_ema.store(models)
        model_ema.copy_to(models)
        print("Switched to EMA weights")
    try:
        yield None
    finally:
        if use_ema:
            model_ema.restore(models)
            print("Restored training weights")


class LitEma(torch.nn.Module):
    def __init__(self, models, decay=0.9999, use_num_updates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.m_name2s_name = {}
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        self.register_buffer(
            "num_updates",
            torch.tensor(0, dtype=torch.int)
            if use_num_updates
            else torch.tensor(-1, dtype=torch.int),
        )

        # Store all model parameters in buffers
        for m in models:
            for name, p in m.named_parameters():
                if p.requires_grad:
                    s_name = name.replace(".", "")
                    self.m_name2s_name.update({f"module.{name}": s_name})
                    self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def reset_num_updates(self):
        del self.num_updates
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.int))

    def forward(self, models):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        # weighting factor decreases as number of steps increases
        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            for m in models:
                m_param = dict(m.named_parameters())
                shadow_params = dict(self.named_buffers())

                for key in m_param:
                    if m_param[key].requires_grad:
                        sname = self.m_name2s_name[key]
                        shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                        # Given initial weights x, new weights y, and weighting factor a, the update is: x = x - a*x + a*y
                        shadow_params[sname].sub_(
                            one_minus_decay * (shadow_params[sname] - m_param[key])
                        )
                    else:
                        assert key not in self.m_name2s_name

    def copy_to(self, models):
        for m in models:
            m_param = dict(m.named_parameters())
            shadow_params = dict(self.named_buffers())
            for key in m_param:
                if m_param[key].requires_grad:
                    m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
                else:
                    assert key not in self.m_name2s_name

    def store(self, models):
        """Save the current parameters for restoring later."""
        self.collected_params = [0] * len(models)
        for i in range(len(models)):
            self.collected_params[i] = [param.clone() for param in models[i].parameters()]

    def restore(self, models):
        """Restore the parameters stored with the `store` method.

        Useful to validate the model with EMA parameters without affecting the original
        optimization process. Store the parameters before the `copy_to` method. After validation
        (or model saving), use this to restore the former parameters.
        """
        for i in range(len(models)):
            for c_param, param in zip(self.collected_params[i], models[i].parameters()):
                param.data.copy_(c_param.data)
