import torch
import torch.nn as nn

from comprx.utils.vae.loss_components import (
    LPIPS,
    NLayerDiscriminator,
    hinge_loss,
    weights_init,
)

__all__ = ["LPIPSWithDiscriminator"]


class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        kl_weight=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        learn_logvar=False,
        num_channels=3,
        ckpt_path=None,
        ignore_keys=[],
    ):

        super().__init__()
        self.learn_logvar = learn_logvar
        self.kl_weight = kl_weight  # Weight assigned to KL regularization term
        self.perceptual_loss = LPIPS().eval()  # Perceptual loss function
        self.perceptual_weight = perceptual_weight  # Weight assigned to perceptual loss
        self.logvar = nn.Parameter(torch.ones(size=()) * 0.0)

        self.discriminator = NLayerDiscriminator(input_nc=num_channels).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.discriminator_weight = disc_weight  # Weight assigned to generator loss

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)

    def init_from_ckpt(self, path, ignore_keys):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        state_dict = {}
        for k in list(sd.keys()):
            if k.startswith("loss"):
                state_dict[".".join(k.split(".")[1:])] = sd[k]
        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del state_dict[k]
        self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path}")

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        posteriors: torch.distributions.Distribution,
        optimizer_idx: int,
        global_step: int,
        weight_dtype: torch.dtype,
        last_layer: nn.Module,
        split: str = "train",
    ):
        bsz = inputs.shape[0]

        if optimizer_idx == 0:
            # Perceptual loss
            # Absolute Error
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            # Perceptual Error (dim = [bsz x 1 x 1 x1])
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
            if weight_dtype == torch.float16:
                nll_loss = (rec_loss / torch.exp(self.logvar) + self.logvar).mean()
            elif weight_dtype == torch.float32:
                nll_loss = (rec_loss / torch.exp(self.logvar) + self.logvar).sum() / bsz
            else:
                raise ValueError("Weight dtype not supported")

            # KL regularization loss
            kl_loss = posteriors.kl()
            kl_loss = kl_loss.sum() / bsz

            # Generator loss (−L_adv(D(E(x)))): Forces discriminator logits to be high when reconstructions are provided
            d_valid = 0 if global_step < self.discriminator_iter_start else 1
            d_weight = torch.tensor(0.0)
            g_loss = torch.tensor(0.0)
            if d_valid:
                logits_fake = self.discriminator(reconstructions.contiguous())
                g_loss = -torch.mean(logits_fake)

                try:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)

            loss = nll_loss + self.kl_weight * kl_loss + d_weight * d_valid * g_loss

            log = {
                f"{split}/total_loss": loss.clone().detach().mean(),
                f"{split}/logvar": self.logvar.detach(),
                f"{split}/kl_loss": kl_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/g_loss": g_loss.detach().mean(),
            }

            return loss, log

        elif optimizer_idx == 1:
            # Discriminator loss (log D_phi(x)): Forces discriminator logits to be high (+1) for inputs and low (-1) for reconstructions
            d_valid = 0 if global_step < self.discriminator_iter_start else 1
            if d_valid:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())

                d_loss = d_valid * hinge_loss(logits_real, logits_fake)

                log = {
                    f"{split}/disc_loss": d_loss.clone().detach().mean(),
                    f"{split}/logits_real": logits_real.detach().mean(),
                    f"{split}/logits_fake": logits_fake.detach().mean(),
                }
            else:
                d_loss = torch.tensor(0.0)
                log = {
                    f"{split}/disc_loss": d_loss.mean(),
                }

            return d_loss, log
