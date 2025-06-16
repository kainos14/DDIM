import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm
from functools import partial


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class TimestepBlock(nn.Module):
    def forward(self, x, emb):
        raise NotImplementedError()


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, TimestepBlock) else layer(x)
        return x


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.norm = Normalize(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attn = QKVAttentionLegacy(num_heads)
        self.proj = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        B, C, T = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        attn = self.attn(qkv)
        return x + self.proj(attn)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv1d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class ResBlock(TimestepBlock):
    def __init__(self, in_ch, emb_ch, out_ch=None, dropout=0.1):
        super().__init__()
        out_ch = out_ch or in_ch
        self.in_layers = nn.Sequential(
            Normalize(in_ch),
            nn.SiLU(),
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_ch, out_ch),
        )
        self.out_layers = nn.Sequential(
            Normalize(out_ch),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv1d(out_ch, out_ch, 3, padding=1)),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype).unsqueeze(-1)
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip(x) + h


class UNetModel(nn.Module):
    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks,
                 attention_resolutions, dropout, channel_mult, num_heads):
        super().__init__()
        time_ch = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_ch),
            nn.SiLU(),
            nn.Linear(time_ch, time_ch),
        )
        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(TimestepEmbedSequential(nn.Conv1d(in_channels, model_channels, 3, padding=1)))
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_ch, mult * model_channels, dropout)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch)))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_ch, dropout=dropout),
            AttentionBlock(ch, num_heads),
            ResBlock(ch, time_ch, dropout=dropout),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_ch, model_channels * mult, dropout)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            Normalize(ch),
            nn.SiLU(),
            zero_module(nn.Conv1d(ch, out_channels, 3, padding=1)),
        )

        self.image_size = image_size
        self.in_channels = in_channels

    def forward(self, x, timesteps):
        t_emb = timestep_embedding(timesteps, self.time_embed[0].in_features)
        emb = self.time_embed(t_emb)
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h_skip = hs.pop()
            if h.shape[2] != h_skip.shape[2]:
                h_skip = F.interpolate(h_skip, size=h.shape[2], mode="nearest")
            h = torch.cat([h, h_skip], dim=1)
            h = module(h, emb)
        return self.out(h)


class DDIM(nn.Module):
    def __init__(self, unet_config, timesteps, ddim_steps, eta=0.0):
        super().__init__()
        self.model = UNetModel(**unet_config)
        self.ddim_steps = ddim_steps
        self.eta = eta
        beta_start = 1e-4
        beta_end = 0.02
        betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1. - alphas_cumprod)))

    def ddim_sample(self, x, t):
        model_output = self.model(x, t)
        pred_x0 = extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - \
                  extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * model_output
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        sigma = self.eta * (1 - extract(self.alphas_cumprod, t, x.shape)).sqrt()
        return pred_x0 + sigma * torch.randn_like(x)

    @torch.no_grad()
    def p_sample_loop_ddim(self, shape):
        x = torch.randn(shape, device=self.alphas_cumprod.device)
        for i in tqdm(reversed(range(0, self.ddim_steps)), desc="DDIM Sampling"):
            t = torch.full((shape[0],), i, dtype=torch.long, device=x.device)
            x = self.ddim_sample(x, t)
        return x

    @torch.no_grad()
    def sample(self, batch_size):
        return self.p_sample_loop_ddim((batch_size, self.model.in_channels, self.model.image_size))

    def forward(self, x, t):
        return self.model(x, t)
