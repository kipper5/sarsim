"""
Stochastic refractory spiking sampler (Architecture 2 substrate).

A network of Buesing-style binary stochastic neurons whose stationary
distribution is a Boltzmann / Markov random field. This module is the
proven mechanism; the terrain energy field that drives it lives in
energy_field.py. Plain PyTorch tensors, not snntorch's LIF layers (a
different, deterministic neuron), so one update serves a single neuron,
a batch, or a full grid of chains on GPU.

  One discrete-time update of a tensor of stochastic refractory neurons.

  refrac : long tensor  - refractory steps remaining; the binary state is
                          z = (refrac > 0).
  u      : float tensor - membrane potential (log-odds), u = b + sum_j W_ij z_j.
  tau    : int          - refractory length; a spike keeps z = 1 for tau steps.

  Returns the updated refrac. Recover state with z = (refrac > 0).

  Order matters:
    1. age first: refrac -> max(refrac - 1, 0), so a neuron leaving
       refractory this step can fire again this step;
    2. a rested neuron (refrac == 0) fires with prob sigmoid(u - log tau).
       The -log tau cancels the fact that a spike lasts tau steps, so the
       long-run activity is sigmoid(u) for any tau. Swap these two and you
       leak one idle step per spike and bias every marginal low.
  """
import math
import torch
import torch.nn.functional as F

def neuron_step(refrac, u, tau, generator=None):
    refrac = torch.clamp(refrac -1, 0.0)
    resting = refrac == 0
    p=torch.sigmoid(u - math.log(tau))
    fire = resting & (torch.rand(refrac.shape, generator=generator, device=refrac.device) < p)
    return torch.where(fire, torch.full_like(refrac, tau), refrac)


# Plus-shaped kernel: each cell's four nearest neighbours.
_NEIGHBOUR_KERNEL = torch.tensor([[[[0., 1., 0.],
                                     [1., 0., 1.],
                                     [0., 1., 0.]]]])


def neighbour_sum(z):
    """Sum of the four nearest-neighbour states for every cell.

    z : (B, H, W) float tensor of binary states (0/1). Returns (B, H, W).
    Zero padding means edge cells simply have fewer neighbours.
    """
    k = _NEIGHBOUR_KERNEL.to(dtype=z.dtype, device=z.device)
    return F.conv2d(z.unsqueeze(1), k, padding=1).squeeze(1)


def checkerboard_masks(h, w, device=None):
    """Red/black boolean masks for the two-colouring of an h x w grid."""
    ii, jj = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    red = ((ii + jj) % 2 == 0).to(device)
    return red, ~red


def sampler_step(refrac, b, coupling, tau, red, black, generator=None):
    for mask in (red, black):
        z = (refrac > 0).to(b.dtype)
        if isinstance(coupling, tuple):
            nb = coupled_neighbour_sum(z, coupling[0], coupling[1])
        else:
            nb = coupling * neighbour_sum(z)
        u = b + nb
        updated = neuron_step(refrac, u, tau, generator)
        refrac = torch.where(mask, updated, refrac)
    return refrac

def run_sampler(b, coupling, tau, n_chains=256, burn_in=1000, readout=2000,
                generator=None):
    """
    Run n_chains parallel samplers and return per-cell spike counts summed
    over the readout window (H, W). Normalise for a belief, or hand the raw
    counts to pdf_common.field_to_pdf for the canonical readout.
    """
    H, W = b.shape
    red, black = checkerboard_masks(H, W, device=b.device)
    refrac = torch.zeros(n_chains, H, W, dtype=torch.long, device=b.device)
    counts = torch.zeros(H, W, dtype=torch.double, device=b.device)
    for t in range(burn_in + readout):
        refrac = sampler_step(refrac, b, coupling, tau, red, black, generator)
        if t >= burn_in:
            counts += (refrac > 0).double().sum(0)
    return counts

def coupled_neighbour_sum(z, Wh, Wv):
    """Weighted sum of neighbour states with per-edge couplings.
    Wh[i, j] weights the edge (i,j)-(i,j+1); Wv[i, j] weights (i,j)-(i+1,j)."""
    out = torch.zeros_like(z)
    out[:, :, :-1] += Wh[:, :-1] * z[:, :, 1:]    # right neighbour
    out[:, :, 1:]  += Wh[:, :-1] * z[:, :, :-1]   # left neighbour
    out[:, :-1, :] += Wv[:-1, :] * z[:, 1:, :]    # down neighbour
    out[:, 1:, :]  += Wv[:-1, :] * z[:, :-1, :]   # up neighbour
    return out