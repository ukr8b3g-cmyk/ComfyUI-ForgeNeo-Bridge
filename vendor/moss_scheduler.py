# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Silver
# Adapted 2026-10-24 for MOSS Shift Scheduler. See LICENSE and THIRD_PARTY_NOTICES.md.

import numpy
import torch


MODES = ("MOSS Shift Scheduler", "MOSS Curve Scheduler", "MOSS Sigma Scheduler")
MODE_ALIASES = {
    "Radiance Shift (Compatible)": MODES[0],
    "MOSS Standard": MODES[1],
    "MOSS DMD": MODES[2],
    "Balanced": MODES[0],
    "Power": MODES[1],
    "Offset": MODES[2],
    "moss_shift_balanced": MODES[0],
    "moss_shift_power": MODES[1],
    "moss_shift_offset": MODES[2],
}


def power_shift_scheduler(model_sampling, steps, power=2.0, midpoint_shift=1.0,
                          discard_penultimate=False, step_offset=0.0):
    total_timesteps = len(model_sampling.sigmas) - 1
    x = numpy.linspace(0, 1, steps, endpoint=False)
    if step_offset:
        x = x * steps / (steps + step_offset)
    x = x**midpoint_shift
    ts_normalized = (1 - x**power)**power
    timesteps = numpy.rint(ts_normalized * total_timesteps)
    sigmas = []
    last_t = -1
    for timestep in timesteps:
        timestep = min(int(timestep), total_timesteps)
        if timestep != last_t:
            sigmas.append(float(model_sampling.sigmas[timestep]))
        last_t = timestep
    sigmas.append(0.0)
    result = torch.FloatTensor(sigmas)
    if discard_penultimate:
        return torch.cat((result[:-2], result[-1:]))
    return result


def radiance_shift_scheduler(model_sampling, steps, power=2.4, midpoint_shift=0.98,
                             discard_penultimate=True):
    total_timesteps = len(model_sampling.sigmas) - 1
    real_steps = steps + 1
    x = numpy.linspace(0, 1, real_steps, endpoint=False)
    x = x**midpoint_shift
    ts_normalized = (1 - x**power)**power
    timesteps = numpy.rint(ts_normalized * total_timesteps)
    sigmas = []
    last_t = -1
    for timestep in timesteps:
        timestep = min(int(timestep), total_timesteps)
        if timestep != last_t:
            sigmas.append(float(model_sampling.sigmas[timestep]))
        last_t = timestep
    sigmas.append(0.0)
    result = torch.FloatTensor(sigmas)
    if discard_penultimate:
        return torch.cat((result[:-2], result[-1:]))
    return result


def moss_standard_scheduler(model_sampling, steps):
    return power_shift_scheduler(model_sampling, steps)


def moss_dmd_scheduler(model_sampling, steps):
    # Radiance uses a steps+1 grid; move toward that grid without duplicating it.
    return power_shift_scheduler(model_sampling, steps, power=2.4, midpoint_shift=0.98,
                                 step_offset=0.75)


MODE_HANDLERS = {
    MODES[0]: radiance_shift_scheduler,
    MODES[1]: moss_standard_scheduler,
    MODES[2]: moss_dmd_scheduler,
}

SCHEDULER_HANDLERS = {
    MODES[0]: radiance_shift_scheduler,
    MODES[1]: moss_standard_scheduler,
    MODES[2]: moss_dmd_scheduler,
}


def get_sigmas(model_sampling, mode, steps, denoise=1.0):
    mode = MODE_ALIASES.get(mode, mode)
    handler = MODE_HANDLERS[mode]
    if denoise <= 0.0:
        return torch.FloatTensor([])
    total_steps = int(steps / denoise) if denoise < 1.0 else steps
    return handler(model_sampling, total_steps)[-(steps + 1):].cpu()
