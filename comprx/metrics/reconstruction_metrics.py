from copy import deepcopy

import clip
import numpy as np
import open_clip
import torch
import torchvision
from torchmetrics import (
    Metric,
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)
from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize


class MSE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, images, reconstructions):
        assert images.shape == reconstructions.shape
        images = images.contiguous()
        reconstructions = reconstructions.contiguous()

        images = images.view(images.shape[0], -1)
        reconstructions = reconstructions.view(reconstructions.shape[0], -1)
        err = ((images - reconstructions) ** 2).mean(-1)

        self.error += err.sum()
        self.total += images.shape[0]

    def compute(self):
        return self.error.float() / self.total


class PSNR(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("psnr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.func = PeakSignalNoiseRatio(data_range=1.0, reduction=None, dim=1)

    def update(self, images, reconstructions):
        assert images.shape == reconstructions.shape
        images = images.contiguous()
        reconstructions = reconstructions.contiguous()

        # Undo normalization and transform images to 0 to 1 range
        images = images.view(images.shape[0], -1)
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)

        # Undo normalization and transform reconstructions to 0 to 1 range
        reconstructions = reconstructions.view(reconstructions.shape[0], -1)
        reconstructions = torch.clamp((reconstructions + 1.0) / 2.0, min=0.0, max=1.0)

        # Compute PSNR
        psnr = self.func(reconstructions, images).sum()

        self.psnr += psnr
        self.total += images.shape[0]

    def compute(self):
        return self.psnr.float() / self.total


class MS_SSIM(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("ms_ssim", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.func = MultiScaleStructuralSimilarityIndexMeasure(reduction="none", data_range=1.0)

    def update(self, images, reconstructions):
        assert images.shape == reconstructions.shape

        # Undo normalization and transform reconstructions to 0 to 1 range
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)

        # Undo normalization and transform reconstructions to 0 to 1 range
        reconstructions = torch.clamp((reconstructions + 1.0) / 2.0, min=0.0, max=1.0)

        # Compute MS-SSIM
        ms_ssim = self.func(reconstructions, images).sum()

        self.ms_ssim += ms_ssim
        self.total += images.shape[0]

    def compute(self):
        return self.ms_ssim.float() / self.total


class FID_Inception(Metric):
    def __init__(self):
        super().__init__()
        self.func = FrechetInceptionDistance(normalize=True)

    def update(self, images, reconstructions):
        assert images.shape == reconstructions.shape

        # Undo normalization and transform reconstructions to 0 to 1 range
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).expand(-1, 3, -1, -1)

        # Undo normalization and transform reconstructions to 0 to 1 range
        reconstructions = torch.clamp((reconstructions + 1.0) / 2.0, min=0.0, max=1.0).expand(
            -1, 3, -1, -1
        )

        # Compute FID
        self.func.update(images, real=True)
        self.func.update(reconstructions, real=False)

    def compute(self):
        return self.func.compute()


class FID_CLIP(Metric):
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    real_features_sum: torch.Tensor
    real_features_cov_sum: torch.Tensor
    real_features_num_samples: torch.Tensor

    fake_features_sum: torch.Tensor
    fake_features_cov_sum: torch.Tensor
    fake_features_num_samples: torch.Tensor

    def __init__(self, version, reset_real_features=True, normalize=False, **kwargs):
        super().__init__(**kwargs)

        if version == "CLIP":
            self.clip, _ = clip.load("ViT-B/32", device="cuda")
            res = self.clip.visual.input_resolution
        elif version == "BiomedCLIP":
            self.clip, _, _ = open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            res = 224
        for _, param in enumerate(self.clip.parameters()):
            param.requires_grad_(False)
        self.clip.eval()
        self.clip_transform = Compose(
            [
                Resize(res, interpolation=3),
                CenterCrop(res),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                ),
            ]
        )

        self.reset_real_features = reset_real_features
        self.normalize = normalize

        num_features = 512
        mx_nb_feats = (num_features, num_features)
        self.add_state(
            "real_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum"
        )
        self.add_state(
            "real_features_cov_sum", torch.zeros(mx_nb_feats).double(), dist_reduce_fx="sum"
        )
        self.add_state("real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.add_state(
            "fake_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum"
        )
        self.add_state(
            "fake_features_cov_sum", torch.zeros(mx_nb_feats).double(), dist_reduce_fx="sum"
        )
        self.add_state("fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

    def update(self, images, reconstructions):
        assert images.shape == reconstructions.shape

        # Undo normalization and transform reconstructions to 0 to 1 range
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images = images.expand(-1, 3, -1, -1)
        # Compute features for real images
        imgs = self.clip_transform(images)
        with torch.no_grad():
            features = self.clip.encode_image(imgs)
        self.orig_dtype = features.dtype
        features = features.double()
        if features.dim() == 1:
            features = features.unsqueeze(0)
        self.real_features_sum += features.sum(dim=0)
        self.real_features_cov_sum += features.t().mm(features)
        self.real_features_num_samples += imgs.shape[0]

        # Undo normalization and transform reconstructions to 0 to 1 range
        reconstructions = torch.clamp((reconstructions + 1.0) / 2.0, min=0.0, max=1.0)
        reconstructions = reconstructions.expand(-1, 3, -1, -1)
        # Compute features for reconstructed images
        imgs = self.clip_transform(reconstructions)
        with torch.no_grad():
            features = self.clip.encode_image(imgs)
        self.orig_dtype = features.dtype
        features = features.double()
        if features.dim() == 1:
            features = features.unsqueeze(0)
        self.fake_features_sum += features.sum(dim=0)
        self.fake_features_cov_sum += features.t().mm(features)
        self.fake_features_num_samples += imgs.shape[0]

    def compute(self):
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = (
            self.real_features_cov_sum
            - self.real_features_num_samples * mean_real.t().mm(mean_real)
        )
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = (
            self.fake_features_cov_sum
            - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        )
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(
            self.orig_dtype
        )

    def reset(self) -> None:
        if not self.reset_real_features:
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_cov_sum = deepcopy(self.real_features_cov_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
        else:
            super().reset()
