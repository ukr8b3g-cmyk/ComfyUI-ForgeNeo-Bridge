# SPDX-License-Identifier: AGPL-3.0-only
# Adapted from Neo-Sampler 925257cf and fixed Forge Neo 710f1e25.
# Bridge: explicit per-run context; no mutable scheduler CTX.
# Port of Forge Neo modules/sd_schedulers.py (neo branch).
# Changes: `shared.opts` / `shared.sd_model` are replaced by the CTX namespace filled in by the sampler node,
# the Forge-only model accessors are replaced by the predictor adapter, and the diffusers
# FlowMatchEulerDiscreteScheduler call is replaced by a line-for-line port of its default code path.
import dataclasses
from math import atan, exp, pi
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
from scipy import stats

from ..vendor.forge import sampling as k_sampling




def to_d(x: torch.Tensor, sigma: float, denoised: torch.Tensor):
    """Converts a denoiser output to a Karras ODE derivative"""
    return (x - denoised) / sigma



@dataclasses.dataclass(frozen=True)
class Scheduler:
    name: str
    label: str
    function: Callable

    default_rho: float = -1.0
    need_inner_model: bool = False
    aliases: list[str] = None


def normal_scheduler(n, sigma_min, sigma_max, inner_model, device, sgm=False, floor=False):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))

    if sgm:
        timesteps = torch.linspace(start, end, n + 1)[:-1]
    else:
        timesteps = torch.linspace(start, end, n)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(inner_model.t_to_sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def simple_scheduler(n, sigma_min, sigma_max, inner_model, device):
    sigs = []
    ss = len(inner_model.sigmas) / n
    for x in range(n):
        sigs += [float(inner_model.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def uniform(n, sigma_min, sigma_max, inner_model, device):
    return inner_model.get_sigmas(n).to(device)


def sgm_uniform(n, sigma_min, sigma_max, inner_model, device):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))
    sigs = [inner_model.t_to_sigma(ts) for ts in torch.linspace(start, end, n + 1)[:-1]]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def _loglinear_interp(t_steps, num_steps):
    """Performs log-linear interpolation of a given array of decreasing numbers"""
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])

    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)

    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys


def get_align_your_steps_sigmas(n, sigma_min, sigma_max, device, *, context):
    """https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html"""

    if context.is_sdxl:
        sigmas = sigmas = [sigma_max, sigma_max / 2.314, sigma_max / 3.875, sigma_max / 6.701, sigma_max / 10.89, sigma_max / 16.954, sigma_max / 26.333, sigma_max / 38.46, sigma_max / 62.457, sigma_max / 129.336, 0.029]
    else:
        # Default to SD 1.5 sigmas.
        sigmas = [sigma_max, sigma_max / 2.257, sigma_max / 3.785, sigma_max / 5.418, sigma_max / 7.749, sigma_max / 10.469, sigma_max / 15.176, sigma_max / 22.415, sigma_max / 36.629, sigma_max / 96.151, 0.029]

    if n != len(sigmas):
        sigmas = np.append(_loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)

    return torch.FloatTensor(sigmas).to(device)


def linear_quadratic(n, sigma_min, sigma_max, device, *, threshold_noise=0.025):
    if n == 1:
        sigma_schedule = [1.0, 0.0]
    else:
        linear_steps = n // 2
        linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        threshold_noise_step_diff = linear_steps - threshold_noise * n
        quadratic_steps = n - linear_steps
        quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
        linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
        const = quadratic_coef * (linear_steps**2)
        quadratic_sigma_schedule = [quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, n)]
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.FloatTensor(sigma_schedule).to(device) * sigma_max


def kl_optimal(n, sigma_min, sigma_max, device):
    alpha_min = torch.arctan(torch.tensor(sigma_min, device=device))
    alpha_max = torch.arctan(torch.tensor(sigma_max, device=device))
    step_indices = torch.arange(n + 1, device=device)
    sigmas = torch.tan(step_indices / n * alpha_min + (1.0 - step_indices / n) * alpha_max)
    return sigmas


def ddim_scheduler(n, sigma_min, sigma_max, inner_model, device):
    sigs = []
    ss = max(len(inner_model.sigmas) // n, 1)
    x = 1
    while x < len(inner_model.sigmas):
        sigs += [float(inner_model.sigmas[x])]
        x += ss
    sigs = sigs[::-1]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def beta_scheduler(n, sigma_min, sigma_max, inner_model, device, *, context):
    """
    Beta scheduler
    Based on "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024)
    """
    alpha = context.opts.beta_dist_alpha
    beta = context.opts.beta_dist_beta

    total_timesteps = len(inner_model.sigmas) - 1
    ts = 1 - np.linspace(0, 1, n, endpoint=False)
    ts = np.rint(stats.beta.ppf(ts, alpha, beta) * total_timesteps)

    sigs = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs += [float(inner_model.sigmas[int(t)])]
        last_t = t
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def turbo_scheduler(n, sigma_min, sigma_max, inner_model, device):
    timesteps = torch.flip(torch.arange(1, n + 1) * float(1000.0 / n) - 1, (0,)).round().long().clip(0, 999)
    sigmas = inner_model.predictor.sigma(timesteps)
    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
    return sigmas.to(device)


def get_bong_tangent_sigmas(steps, slope, pivot, start, end):
    smax = ((2 / pi) * atan(-slope * (0 - pivot)) + 1) / 2
    smin = ((2 / pi) * atan(-slope * ((steps - 1) - pivot)) + 1) / 2

    srange = smax - smin
    sscale = start - end

    sigmas = [((((2 / pi) * atan(-slope * (x - pivot)) + 1) / 2) - smin) * (1 / srange) * sscale + end for x in range(steps)]

    return sigmas


def bong_tangent_scheduler(n, sigma_min, sigma_max, device, *, start=1.0, middle=0.5, end=0.0, pivot_1=0.6, pivot_2=0.6, slope_1=0.2, slope_2=0.2, pad=False):
    """https://github.com/ClownsharkBatwing/RES4LYF/blob/main/sigmas.py#L4076"""
    n += 2

    midpoint = int((n * pivot_1 + n * pivot_2) / 2)
    pivot_1 = int(n * pivot_1)
    pivot_2 = int(n * pivot_2)

    slope_1 = slope_1 / (n / 40)
    slope_2 = slope_2 / (n / 40)

    stage_2_len = n - midpoint
    stage_1_len = n - stage_2_len

    tan_sigmas_1 = get_bong_tangent_sigmas(stage_1_len, slope_1, pivot_1, start, middle)
    tan_sigmas_2 = get_bong_tangent_sigmas(stage_2_len, slope_2, pivot_2 - stage_1_len, middle, end)

    tan_sigmas_1 = tan_sigmas_1[:-1]
    if pad:
        tan_sigmas_2 = tan_sigmas_2 + [0]

    tan_sigmas = torch.tensor(tan_sigmas_1 + tan_sigmas_2)

    return tan_sigmas.to(device)


def flow_match_euler_discrete_scheduler(n, sigma_min, sigma_max, inner_model, device, *, context):
    # Neo calls diffusers.FlowMatchEulerDiscreteScheduler.from_config(config).set_timesteps(n, device=device, mu=0.0)
    # with shift = predictor.shift and every other option at Neo's default (all False).
    # Below is diffusers' code path for exactly that configuration.
    opts = context.opts
    if any([opts.use_dynamic_shifting, opts.invert_sigmas, opts.use_karras_sigmas, opts.use_exponential_sigmas, opts.use_beta_sigmas]):
        raise NotImplementedError("FlowMatchEulerDiscrete: only Neo's default settings are supported")

    num_train_timesteps = 1000
    shift = getattr(inner_model.predictor, "shift", 1.0)

    # __init__
    timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
    timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
    sigmas = timesteps / num_train_timesteps
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    sigmas = sigmas.to("cpu")
    _sigma_min = sigmas[-1].item()
    _sigma_max = sigmas[0].item()

    # set_timesteps(n, device, mu=0.0)
    timesteps = np.linspace(_sigma_max * num_train_timesteps, _sigma_min * num_train_timesteps, n)
    sigmas = timesteps / num_train_timesteps
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
    sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

    return torch.FloatTensor(sigmas).to(device)


def generalized_time_snr_shift(t: torch.Tensor, mu: float, sigma: float) -> float:
    return exp(mu) / (exp(mu) + (1 / t - 1) ** sigma)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps


def flux2_scheduler(n: int, width: int, height: int, sigma_min, sigma_max, device):
    # https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_flux.py
    seq_len = width * height / (16 * 16)
    sigmas = get_schedule(n, round(seq_len))
    return torch.FloatTensor(sigmas).to(device)


all_schedulers = [
    Scheduler("automatic", "Automatic", None),
    Scheduler("karras", "Karras", k_sampling.get_sigmas_karras, default_rho=7.0),
    Scheduler("exponential", "Exponential", k_sampling.get_sigmas_exponential),
    Scheduler("polyexponential", "Polyexponential", k_sampling.get_sigmas_polyexponential, default_rho=1.0),
    Scheduler("normal", "Normal", normal_scheduler, need_inner_model=True),
    Scheduler("simple", "Simple", simple_scheduler, need_inner_model=True),
    Scheduler("uniform", "Uniform", uniform, need_inner_model=True),
    Scheduler("sgm_uniform", "SGM Uniform", sgm_uniform, need_inner_model=True, aliases=["SGMUniform"]),
    Scheduler("linear_quadratic", "Linear Quadratic", linear_quadratic),
    Scheduler("kl_optimal", "KL Optimal", kl_optimal),
    Scheduler("ddim", "DDIM", ddim_scheduler, need_inner_model=True),
    Scheduler("align_your_steps", "Align Your Steps", get_align_your_steps_sigmas),
    Scheduler("beta", "Beta", beta_scheduler, need_inner_model=True),
    Scheduler("turbo", "Turbo", turbo_scheduler, need_inner_model=True),
    Scheduler("bong_tangent", "Bong Tangent", bong_tangent_scheduler),
    Scheduler("flow_match", "FlowMatchEulerDiscrete", flow_match_euler_discrete_scheduler, need_inner_model=True),
    Scheduler("flux2", "Flux2", flux2_scheduler),
]

schedulers = list(all_schedulers)

schedulers_map = {**{x.name: x for x in schedulers}, **{x.label: x for x in schedulers}}

# Public execution adapter: UI names are parsed separately, never silently substituted.
import inspect
from .spec import BridgeError, SCHEDULERS

SAMPLER_OPTIONS = {
    'euler': {}, 'euler_ancestral': {'uses_ensd': True}, 'er_sde': {},
    'dpmpp_2m': {'scheduler': 'karras'},
    'dpmpp_2m_sde': {'scheduler': 'exponential', 'brownian_noise': True},
    'dpm_2': {'scheduler': 'karras', 'discard_next_to_last_sigma': True, 'second_order': True},
    'heun': {'second_order': True},
    'lcm': {}, 'lms': {}, 'res_multistep': {},
    'dpmpp_sde': {'scheduler': 'karras', 'second_order': True, 'brownian_noise': True},
    'dpmpp_3m_sde': {'scheduler': 'exponential', 'discard_next_to_last_sigma': True, 'brownian_noise': True},
    # Core solver, Forge schedule. DDIM's timestep/Eta implementation is not ported.
    'ddim': {}, 'unipc': {'discard_next_to_last_sigma': True},
    'euler_cfg_pp': {'requires_uncond_at_cfg1': True},
    'euler_ancestral_cfg_pp': {'uses_ensd': True, 'requires_uncond_at_cfg1': True},
    'dpmpp_2m_cfg_pp': {'scheduler': 'karras', 'requires_uncond_at_cfg1': True},
}

class Predictor:
    def __init__(self, model_sampling, prediction_type):
        self.ms = model_sampling
        self.prediction_type = 'const' if prediction_type == 'flow' else prediction_type

    @property
    def sigmas(self): return self.ms.sigmas
    @property
    def shift(self): return getattr(self.ms, 'shift', 1.0)
    def sigma(self, t): return self.ms.sigma(t)
    def timestep(self, s): return self.ms.timestep(s)
    def percent_to_sigma(self, p): return self.ms.percent_to_sigma(p)
    def noise_scaling(self, sigma, noise, latent, max_denoise=False):
        # Forge AbstractPrediction.noise_scaling, not KSAMPLER's auto max_denoise.
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        if self.prediction_type == 'const': return sigma * noise + (1.0 - sigma) * latent
        return noise * (torch.sqrt(1.0 + sigma**2.0) if max_denoise else sigma) + latent

class Linker:
    def __init__(self, predictor): self.predictor = predictor
    @property
    def sigmas(self): return self.predictor.sigmas
    def sigma_to_t(self, s): return self.predictor.timestep(s)
    def t_to_sigma(self, t): return self.predictor.sigma(t)
    def get_sigmas(self, n):
        t = torch.linspace(len(self.sigmas)-1, 0, n, device=self.sigmas.device)
        return k_sampling.append_zero(self.t_to_sigma(t))


def step_plan(config):
    s = config['sampling']
    if not s.get('adjustments',True):
        n,d = s['steps'],s['denoise']
        if d <= 0:return 0,0,None
        total = n if d > .9999 else int(n/d)
        return total,n,total-n if total>n else None
    if config['mode'] == 'txt2img': return s['steps'], s['steps'], None
    d, n = s['denoise'], s['steps']
    if s['img2img_step_mode'] == 'exact_steps':
        if d == 0: raise BridgeError('UNSUPPORTED_COMBINATION', 'Exact-step zero-denoise cannot form a reference schedule')
        scheduled, t_enc = int(n / min(d, .999)), n - 1
    else:
        scheduled, t_enc = n, int(min(d, .999) * n)
    return scheduled, t_enc + 1, scheduled - t_enc - 1


def core_names(sampling):
    return ({'unipc':'uni_pc'}.get(sampling['sampler'],sampling['sampler']),
            {'automatic':'normal','uniform':'normal','ddim':'ddim_uniform'}.get(sampling['scheduler'],sampling['scheduler']))


def core_sigmas(config, model):
    import comfy.samplers as core
    s = config['sampling']
    sampler,name = core_names(s)
    if sampler not in core.SAMPLER_NAMES or name not in core.SCHEDULER_NAMES:
        raise BridgeError('COMFY_UNSUPPORTED_SAMPLING',
                          f'ComfyUI does not support {sampler} / {name}. Enable Sampling adjustments for Forge-only schedules, or choose a ComfyUI schedule.')
    runner = core.KSampler(model,s['steps'],model.load_device,sampler=sampler,scheduler=name,denoise=s['denoise'])
    sigmas = runner.sigmas.cpu()
    scheduled,actual,start = step_plan(config)
    return sigmas, {'resolved_scheduler':name,'scheduled_steps':scheduled,'launch_steps':actual,
                    'total_denoiser_calls':actual*(2 if SAMPLER_OPTIONS[s['sampler']].get('second_order') else 1),
                    'discard_applied':sampler in core.KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS,
                    'full_sigmas':sigmas.tolist(),'slice_start':start,'backend':'comfy'}


def generate_sigmas(config, predictor):
    s = config['sampling']
    scheduled, actual, start = step_plan(config)
    options = SAMPLER_OPTIONS.get(s['sampler'])
    if options is None: raise BridgeError('UNSUPPORTED_COMBINATION', 'Sampler is registered but not implemented')
    discard = s['discard_penultimate_requested'] or options.get('discard_next_to_last_sigma', False)
    n = scheduled + int(discard)
    name = s['scheduler']
    if name not in SCHEDULERS: raise BridgeError('UNSUPPORTED_COMBINATION', 'Scheduler has no adapter')
    if name == 'automatic': name = options.get('scheduler', 'normal' if config['family'] == 'anima' else 'uniform')
    sch = schedulers_map[name]
    linker = Linker(predictor)
    sigma_min = float(linker.sigmas[0]) if s['sigma_min'] is None else s['sigma_min']
    sigma_max = float(linker.sigmas[-1]) if s['sigma_max'] is None else s['sigma_max']
    if sigma_min < 0 or sigma_max <= sigma_min: raise BridgeError('INVALID_SPEC', 'Invalid sigma range')
    context = SimpleNamespace(is_sdxl=config['family'] == 'sdxl', opts=SimpleNamespace(
        beta_dist_alpha=s['beta_alpha'], beta_dist_beta=s['beta_beta'], **s['flow_match_options']))
    kwargs = {'n': n, 'sigma_min': sigma_min, 'sigma_max': sigma_max, 'device': torch.device('cpu')}
    if sch.need_inner_model: kwargs['inner_model'] = linker
    if sch.default_rho != -1 and s['rho'] is not None: kwargs['rho'] = s['rho']
    if 'context' in inspect.signature(sch.function).parameters: kwargs['context'] = context
    sigmas = sch.function(**kwargs)
    if discard: sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    full = sigmas.detach().cpu().float()
    selected = full if start is None else full[start:]
    if selected.ndim != 1 or len(selected) < 2 or not torch.isfinite(selected).all() or selected[-1] < 0 or not torch.all(selected[:-1] > selected[1:]):
        raise BridgeError('UNSUPPORTED_COMBINATION', 'Reference schedule is not a finite strictly descending non-negative vector')
    return selected, {'resolved_scheduler': name, 'scheduled_steps': scheduled, 'launch_steps': actual,
                      'total_denoiser_calls': actual * (2 if options.get('second_order') else 1),
                      'discard_applied': bool(discard), 'full_sigmas': full.tolist(), 'slice_start': start}
