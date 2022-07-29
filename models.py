# -*- coding: utf-8 -*-
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import numpy as np

import commons
import modules
import attentions
import monotonic_align

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
from frame_prior_network import ConformerBlock, VariancePredictor, ResidualConnectionModule
from symbols import ctc_symbols


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()
        self.winlen = 1024
        self.hoplen = 256
        self.sr = 44100

    def LR(self, x, notepitch, duration, x_lengths):
        output = list()
        frame_pitch = list()
        mel_len = list()
        x = torch.transpose(x, 1, -1)
        frame_lengths = list()

        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            frame_lengths.append(expanded.shape[0])

        for batch, expand_target in zip(notepitch, duration):
            expanded_pitch = self.expand_pitch(batch, expand_target)
            frame_pitch.append(expanded_pitch)

        max_len = max(frame_lengths)
        output_padded = torch.FloatTensor(x.size(0), max_len, x.size(2))
        output_padded.zero_()
        frame_pitch_padded = torch.FloatTensor(notepitch.size(0), max_len)
        frame_pitch_padded.zero_()
        for i in range(output_padded.size(0)):
            output_padded[i, :frame_lengths[i], :] = output[i]
        for i in range(frame_pitch_padded.size(0)):
            length = len(frame_pitch[i])
            frame_pitch[i].extend([0] * (max_len - length))
            frame_pitch_tensor = torch.LongTensor(frame_pitch[i])
            frame_pitch_padded[i] = frame_pitch_tensor
        output_padded = torch.transpose(output_padded, 1, -1)

        return output_padded, frame_pitch_padded, torch.LongTensor(frame_lengths)

    def expand_pitch(self, batch, predicted):
        out = list()
        predicted = predicted.squeeze()
        for i, vec in enumerate(batch):
            duration = predicted[i].item()
            if self.sr * duration - self.winlen > 0:
                expand_size = max((self.sr * duration - self.winlen) / self.hoplen, 1)
            elif duration == 0:
                expand_size = 0
            else:
                expand_size = 1
            vec_expand = vec.expand(max(int(expand_size), 0),1).squeeze(1).cpu().numpy()
            out.extend(vec_expand)

        torch.LongTensor(out).to(vec.device)
        return out

    def expand(self, batch, predicted):
        out = list()
        predicted = predicted.squeeze()
        for i, vec in enumerate(batch):

            duration = predicted[i].item()
            if self.sr * duration - self.winlen > 0:
                expand_size = max((self.sr * duration - self.winlen)/self.hoplen, 1)
            elif duration == 0:
                expand_size = 0
            else:
                expand_size = 1
            vec_expand = vec.expand(max(int(expand_size), 0), -1)
            out.append(vec_expand)

        out = torch.cat(out, 0)
        return out

    def forward(self, x, notepitch, duration, x_lengths):

        notepitch = torch.detach(notepitch)
        output, frame_pitch, x_lengths = self.LR(x, notepitch, duration, x_lengths)
        return output, frame_pitch, x_lengths


class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    super().__init__()
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.log_flow = modules.Log()
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.flows.append(modules.Flip())

    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2))
    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(modules.Flip())

    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

  def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x)
    x = self.pre(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
      flows = self.flows
      assert w is not None

      logdet_tot_q = 0
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = torch.split(z_q, [1, 1], 1)
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
      logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
      return nll + logq # [b]
    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]] # remove a useless vflow
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
      for flow in flows:
        z = flow(z, x_mask, g=x, reverse=reverse)
      z0, z1 = torch.split(z, [1, 1], 1)
      logw = z0
      return logw


class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    super().__init__()

    self.in_channels = in_channels + 1
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels + 1, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

  def forward(self, x, x_mask, notedur, g=None):
    x = torch.detach(x)
    notedur = torch.detach(notedur)
    notedur = notedur.unsqueeze(1)
    x = torch.cat((x, notedur), 1)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask


class PhonemesPredictor(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)

        self.phonemes_predictor = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            2,
            kernel_size,
            p_dropout)
        self.linear1 = nn.Linear(hidden_channels, 62)

    def forward(self, x, x_mask):
        phonemes_embedding = self.phonemes_predictor(x * x_mask, x_mask)
        print("x_size:", x.size())
        x1 = self.linear1(phonemes_embedding.transpose(1, 2))
        x1 = x1.log_softmax(2)
        print("phonemes_embedding size:", x1.size())
        return x1.transpose(0, 1)


class PitchPredictor(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.n_vocab = n_vocab  # 音素的个数，中文和英文不同
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(121, hidden_channels)

        self.pitch_net = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, x, x_mask):
        pitch_embedding = self.pitch_net(x * x_mask, x_mask)
        pitch_embedding = pitch_embedding * x_mask
        pred_pitch = self.proj(pitch_embedding)
        return pred_pitch, pitch_embedding


class TextEncoder(nn.Module):
  def __init__(self,
      n_vocab,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout):
    super().__init__()
    self.n_vocab = n_vocab # 音素的个数，中文和英文不同
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.emb = nn.Embedding(n_vocab, hidden_channels)

    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
    self.pitch_embedding = nn.Embedding(
        121, 192
    )
    self.duration_embedding = nn.Embedding(
        1200, 192
    )

  def forward(self, x, x_lengths, notepitch, notedur):
    notepitch = self.pitch_embedding(notepitch)
    notedur = ((notedur * 172) - 3).clamp_min_(0).long()
    notedur = self.duration_embedding(notedur)
    x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
    x = x + notedur + notepitch
    x = torch.transpose(x, 1, -1)  # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.encoder(x * x_mask, x_mask)

    return x, x_mask


class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x


class PosteriorEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Projection(nn.Module):
    def __init__(self,
                 hidden_channels,
                 out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_mask):
        stats = self.proj(x) * x_mask
        m_p, logs_p = torch.split(stats, self.out_channels, dim=1)
        return m_p, logs_p


class FramePriorBlock(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=nn.Conv1d(
                    in_channels,
                    filter_channels,
                    kernel_size,
                    padding=kernel_size // 2)
            ),
            nn.ReLU(),
            modules.LayerNorm(filter_channels),
            nn.Dropout(p_dropout)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)


class FramePriorNet(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()

        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(121, hidden_channels)

        self.fft_block = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            4,
            kernel_size,
            p_dropout)

    def forward(self, x_frame, pitch_embedding, x_mask):
        x = x_frame + pitch_embedding
        x = self.fft_block(x * x_mask, x_mask)
        x = x.transpose(1, 2)
        return x


class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self,
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock,
    resblock_kernel_sizes,
    resblock_dilation_sizes,
    upsample_rates,
    upsample_initial_channel,
    upsample_kernel_sizes,
    n_speakers=0,
    gin_channels=0,
    use_sdp=False,
    **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.n_speakers = n_speakers
    self.gin_channels = gin_channels

    self.use_sdp = use_sdp
    self.enc_p = TextEncoder(n_vocab,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout)
    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)
    self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)
    self.project = Projection(hidden_channels, inter_channels)
    self.lr = LengthRegulator()
    self.frame_prior_net = FramePriorNet(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
    # self.pitch_net = VariancePredictor()
    self.pitch_net = PitchPredictor(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
    self.phonemes_predictor = PhonemesPredictor(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
    self.ctc_loss = nn.CTCLoss(len(ctc_symbols)-1, reduction='mean')
    if n_speakers > 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels)

  def forward(self, phonemes, phonemes_lengths, notepitch, notedur, phndur, spec, spec_lengths, sid=None):
    g = None
    x, x_mask = self.enc_p(phonemes, phonemes_lengths, notepitch, notedur)
    w = phndur.unsqueeze(1)
    logw_ = w * x_mask
    logw = self.dp(x, x_mask, notedur, g=g)
    logw = torch.mul(logw.squeeze(1), notedur).unsqueeze(1)
    l_loss = torch.sum((logw - logw_)**2, [1, 2])
    x_mask_sum = torch.sum(x_mask)
    l_length = l_loss / x_mask_sum
    x_frame, frame_pitch, x_lengths = self.lr(x, notepitch, phndur, phonemes_lengths)
    x_frame = x_frame.to(x.device)
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x_frame.size(2)), 1).to(x.dtype)  # 更新x_mask矩阵
    x_mask = x_mask.to(x.device)
    max_len = x_frame.size(2)
    d_model = x_frame.size(1)
    batch_size = x_frame.size(0)
    pe = torch.zeros(batch_size, max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)
    pe = pe.transpose(1, 2).to(x_frame.device)
    x_frame = x_frame + pe
    pred_pitch, pitch_embedding = self.pitch_net(x_frame, x_mask)
    lf0 = torch.unsqueeze(pred_pitch, -1)
    gt_lf0 = torch.log(440 * (2 ** ((frame_pitch - 69) / 12)))
    gt_lf0 = gt_lf0.to(x.device)
    x_mask_sum = torch.sum(x_mask)
    lf0 = lf0.squeeze()
    l_pitch = torch.sum((gt_lf0 - lf0) ** 2, 1) / x_mask_sum
    frame_pitch = frame_pitch + 1e-6
    frame_pitch = torch.log(frame_pitch)
    frame_pitch = torch.unsqueeze(frame_pitch, -1)
    frame_pitch = frame_pitch.to(x.device)
    x_frame = self.frame_prior_net(x_frame, pitch_embedding, x_mask)
    x_frame = x_frame.transpose(1, 2)
    m_p, logs_p = self.project(x_frame, x_mask)
    z, m_q, logs_q, y_mask = self.enc_q(spec, spec_lengths, g=g)
    log_probs = self.phonemes_predictor(z, y_mask)
    ctc_loss = self.ctc_loss(log_probs, phonemes, spec_lengths, phonemes_lengths)
    z_p = self.flow(z, y_mask, g=g)
    with torch.no_grad():
      # negative cross-entropy
      s_p_sq_r = torch.exp(-2 * logs_p) # [b, d, t]
      neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True) # [b, 1, t_s]
      neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r)) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True) # [b, 1, t_s]
      neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
      attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
      attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
    z_slice, ids_slice = commons.rand_slice_segments(z, spec_lengths, self.segment_size)
    o = self.dec(z_slice, g=g)
    return o, l_length, l_pitch, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), ctc_loss

  def infer(self, phonemes, phonemes_lengths, notepitch,  notedur,
            sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
    g = None
    x, x_mask = self.enc_p(phonemes, phonemes_lengths, notepitch, notedur)

    logw = self.dp(x, x_mask, notedur, g=g)
    logw = torch.mul(logw.squeeze(1), notedur).unsqueeze(1)
    w = logw * x_mask * length_scale
    x_frame, frame_pitch, x_lengths = self.lr(x, notepitch, w, phonemes_lengths)
    x_frame = x_frame.to(x.device)
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x_frame.size(2)), 1).to(x.dtype)
    x_mask = x_mask.to(x.device)
    max_len = x_frame.size(2)
    d_model = x_frame.size(1)
    batch_size = x_frame.size(0)
    pe = torch.zeros(batch_size, max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)
    pe = pe.transpose(1, 2).to(x_frame.device)
    x_frame = x_frame + pe
    pred_pitch, pitch_embedding = self.pitch_net(x_frame, x_mask)
    lf0 = torch.unsqueeze(pred_pitch, -1)
    f0 = torch.exp(lf0)
    f0 = f0.to(x.device)
    gt_f0 = (440 * (2 ** ((frame_pitch - 69) / 12)))
    gt_f0 = gt_f0.to(x.device)
    x_mask_sum = torch.sum(x_mask)
    f0 = f0.squeeze()
    l_pitch = torch.sum(abs(gt_f0 - f0), 1) / x_mask_sum
    l_pitch = torch.sum(l_pitch.float())
    x_frame = self.frame_prior_net(x_frame, pitch_embedding, x_mask)
    x_frame = x_frame.transpose(1, 2)
    m_p, logs_p = self.project(x_frame, x_mask)
    w_ceil = torch.ceil(w)
    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, x_mask, g=g, reverse=True)
    o = self.dec((z * x_mask)[:, :, :max_len], g=g)
    return o, x_mask, (z, z_p, m_p, logs_p)

  def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    g_src = self.emb_g(sid_src).unsqueeze(-1)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
    z_p = self.flow(z, y_mask, g=g_src)
    z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
    o_hat = self.dec(z_hat * y_mask, g=g_tgt)
    return o_hat, y_mask, (z, z_p, z_hat)

