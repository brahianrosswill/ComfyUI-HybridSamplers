import math
import torch
import comfy.samplers as cs
import comfy.k_diffusion.sampling as kdiff_sampling

# -------------------------------------------------------------------
# Custom Sampler Implementations (wrappers around base samplers)
# -------------------------------------------------------------------

def sampler_adaptive_euler(model, x, sigmas, extra_args=None, callback=None, disable=False, params=None):
    print("[HybridSamplers] Using custom sampler: AdaptiveEuler")
    def cb(state):
        xi = state["x"]
        sigma = state["sigma"]
        scale = 1.0 - 0.1 * torch.tanh(sigma)
        return xi * scale
    return kdiff_sampling.sample_euler(model, x, sigmas, extra_args=extra_args, callback=cb, disable=disable)

def sampler_adaptive_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=False, params=None):
    print("[HybridSamplers] Using custom sampler: AdaptiveEulerAncestral")
    def cb(state):
        xi = state["x"]
        sigma = state["sigma"]
        scale = 1.0 - 0.1 * torch.tanh(sigma)
        return xi * scale
    return kdiff_sampling.sample_euler_ancestral(model, x, sigmas, extra_args=extra_args, callback=cb, disable=disable)

def sampler_dynamic_langevin(model, x, sigmas, extra_args=None, callback=None, disable=False, params=None):
    print("[HybridSamplers] Using custom sampler: DynamicLangevin")
    def cb(state):
        xi = state["x"]
        sigma = state["sigma"]
        noise = torch.randn_like(xi) * sigma * 0.01
        return xi + noise
    return kdiff_sampling.sample_euler(model, x, sigmas, extra_args=extra_args, callback=cb, disable=disable)

def sampler_stochastic_rk(model, x, sigmas, extra_args=None, callback=None, disable=False, params=None):
    print("[HybridSamplers] Using custom sampler: StochasticRungeKutta")
    def cb(state):
        xi = state["x"]
        sigma = state["sigma"]
        jitter = torch.randn_like(xi) * sigma * 0.005
        return xi + jitter
    return kdiff_sampling.sample_heun(model, x, sigmas, extra_args=extra_args, callback=cb, disable=disable)

def sampler_temporal_sampling(model, x, sigmas, extra_args=None, callback=None, disable=False, params=None):
    print("[HybridSamplers] Using custom sampler: TemporalSampling")
    last = {"x": None}
    def cb(state):
        xi = state["x"]
        if last["x"] is None:
            last["x"] = xi
            return xi
        blended = 0.7 * xi + 0.3 * last["x"]
        last["x"] = xi
        return blended
    return kdiff_sampling.sample_euler(model, x, sigmas, extra_args=extra_args, callback=cb, disable=disable)

def sampler_spatial_sampling(model, x, sigmas, extra_args=None, callback=None, disable=False, params=None):
    print("[HybridSamplers] Using custom sampler: SpatialSampling")
    def cb(state):
        xi = state["x"]
        noise = torch.randn_like(xi) * 0.002
        return xi + noise
    return kdiff_sampling.sample_euler(model, x, sigmas, extra_args=extra_args, callback=cb, disable=disable)

def sampler_quantized(model, x, sigmas, extra_args=None, callback=None, disable=False, params=None):
    print("[HybridSamplers] Using custom sampler: Quantized")
    def cb(state):
        xi = state["x"]
        return torch.round(xi * 128) / 128
    return kdiff_sampling.sample_euler(model, x, sigmas, extra_args=extra_args, callback=cb, disable=disable)

def sampler_anisotropic(model, x, sigmas, extra_args=None, callback=None, disable=False, params=None):
    print("[HybridSamplers] Using custom sampler: Anisotropic")
    def cb(state):
        xi = state["x"]
        noise = torch.randn_like(xi)
        noise[:, :, ::2, :] *= 0.5  # attenuate on one axis
        return xi + 0.01 * noise
    return kdiff_sampling.sample_euler(model, x, sigmas, extra_args=extra_args, callback=cb, disable=disable)

def sampler_multidimensional(model, x, sigmas, extra_args=None, callback=None, disable=False, params=None):
    print("[HybridSamplers] Using custom sampler: MultiDimensional")
    def cb(state):
        return state["x"]  # no-op
    out1 = kdiff_sampling.sample_euler(model, x, sigmas, extra_args=extra_args, callback=cb, disable=disable)
    out2 = kdiff_sampling.sample_heun(model, x, sigmas, extra_args=extra_args, callback=cb, disable=disable)
    return 0.5 * (out1 + out2)

CUSTOM_SAMPLER_IMPL = {
    "AdaptiveEuler": {"fn": sampler_adaptive_euler},
    "AdaptiveEulerAncestral": {"fn": sampler_adaptive_euler_ancestral},
    "DynamicLangevin": {"fn": sampler_dynamic_langevin},
    "StochasticRungeKutta": {"fn": sampler_stochastic_rk},
    "TemporalSampling": {"fn": sampler_temporal_sampling},
    "SpatialSampling": {"fn": sampler_spatial_sampling},
    "Quantized": {"fn": sampler_quantized},
    "Anisotropic": {"fn": sampler_anisotropic},
    "MultiDimensional": {"fn": sampler_multidimensional},
}

# Register custom samplers
for name, entry in CUSTOM_SAMPLER_IMPL.items():
    if name not in cs.KSampler.SAMPLERS:
        cs.KSampler.SAMPLERS.append(name)
    attr = f"sample_{name}"
    if not hasattr(kdiff_sampling, attr):
        setattr(kdiff_sampling, attr, entry["fn"])

# -------------------------------------------------------------------
# Custom Scheduler Implementations (sigma transformers only)
# -------------------------------------------------------------------

def sched_adaptive_time(model_sampling, sigmas, steps, params):
    print("[HybridSamplers] Using custom scheduler: AdaptiveTime")
    time_scale = float(params.get("time_scale", 1.2))
    decay_rate = float(params.get("decay_rate", 0.05))
    return torch.tensor([s / (1.0 + decay_rate * time_scale) for s in sigmas], device=sigmas.device)

def sched_dynamic_schedule(model_sampling, sigmas, steps, params):
    print("[HybridSamplers] Using custom scheduler: DynamicSchedule")
    schedule_type = params.get("schedule_type", "cosine")
    interpolation = float(params.get("interpolation", 0.7))
    factor = 0.9 + 0.1*math.cos(interpolation) if schedule_type == "cosine" else 1.0 - 0.1*interpolation
    return sigmas * factor

def sched_variable_step(model_sampling, sigmas, steps, params):
    print("[HybridSamplers] Using custom scheduler: VariableStep")
    min_step = float(params.get("min_step", 0.01))
    max_step = float(params.get("max_step", 0.1))
    factor = 1.0 - (min_step + max_step) / 2.0
    return sigmas * factor

def sched_progressive_decay(model_sampling, sigmas, steps, params):
    print("[HybridSamplers] Using custom scheduler: ProgressiveDecay")
    decay_factor = float(params.get("decay_factor", 0.8))
    step_interval = int(params.get("step_interval", 5))
    new = []
    for i, s in enumerate(sigmas):
        decay = decay_factor ** (i // step_interval)
        new.append(s * decay)
    return torch.tensor(new, device=sigmas.device)

def sched_adaptive_exponential(model_sampling, sigmas, steps, params):
    print("[HybridSamplers] Using custom scheduler: AdaptiveExponential")
    growth_rate = float(params.get("growth_rate", 0.02))
    saturation_point = float(params.get("saturation_point", 0.9))
    return sigmas * (1.0 + growth_rate * (1.0 - saturation_point))

def sched_fractal_time(model_sampling, sigmas, steps, params):
    print("[HybridSamplers] Using custom scheduler: FractalTime")
    fractal_dimension = float(params.get("fractal_dimension", 1.3))
    factor = 1.0 / (1.0 + (fractal_dimension - 1.0) * 0.1)
    return sigmas * factor

def sched_temporal_gradient(model_sampling, sigmas, steps, params):
    print("[HybridSamplers] Using custom scheduler: TemporalGradient")
    grad = float(params.get("gradient_magnitude", 0.5))
    smooth = float(params.get("smoothing_factor", 0.3))
    factor = 1.0 - grad * smooth * 0.1
    return sigmas * factor

def sched_memory_aware(model_sampling, sigmas, steps, params):
    print("[HybridSamplers] Using custom scheduler: MemoryAware")
    retention = float(params.get("retention_factor", 0.9))
    return sigmas * retention

def sched_multi_objective(model_sampling, sigmas, steps, params):
    print("[HybridSamplers] Using custom scheduler: MultiObjective")
    weights = params.get("weights", [0.6, 0.4])
    avg = sum(weights) / len(weights) if weights else 1.0
    return sigmas * avg

def sched_resource_constrained(model_sampling, sigmas, steps, params):
    print("[HybridSamplers] Using custom scheduler: ResourceConstrained")
    eff = float(params.get("efficiency_target", 0.75))
    return sigmas * eff

def sched_dynamic_window(model_sampling, sigmas, steps, params):
    print("[HybridSamplers] Using custom scheduler: DynamicWindow")
    adaptive = bool(params.get("adaptive_window", True))
    factor = 1.0 - (0.1 if adaptive else 0.0)
    return sigmas * factor

def sched_multi_agent(model_sampling, sigmas, steps, params):
    print("[HybridSamplers] Using custom scheduler: MultiAgent")
    coord = float(params.get("coordination_factor", 0.6))
    factor = 1.0 - coord * 0.05
    return sigmas * factor

CUSTOM_SCHEDULER_IMPL = {
    "AdaptiveTime": {"fn": sched_adaptive_time},
    "DynamicSchedule": {"fn": sched_dynamic_schedule},
    "VariableStep": {"fn": sched_variable_step},
    "ProgressiveDecay": {"fn": sched_progressive_decay},
    "AdaptiveExponential": {"fn": sched_adaptive_exponential},
    "FractalTime": {"fn": sched_fractal_time},
    "TemporalGradient": {"fn": sched_temporal_gradient},
    "MemoryAware": {"fn": sched_memory_aware},
    "MultiObjective": {"fn": sched_multi_objective},
    "ResourceConstrained": {"fn": sched_resource_constrained},
    "DynamicWindow": {"fn": sched_dynamic_window},
    "MultiAgent": {"fn": sched_multi_agent},
}

# Register custom schedulers
for name in CUSTOM_SCHEDULER_IMPL.keys():
    if name not in cs.KSampler.SCHEDULERS:
        cs.KSampler.SCHEDULERS.append(name)

# -------------------------------------------------------------------
# Patch calculate_sigmas to handle custom schedulers
# -------------------------------------------------------------------

_original_calculate_sigmas = cs.calculate_sigmas

def calculate_sigmas_patched(model_sampling, scheduler_name, steps):
    if scheduler_name in CUSTOM_SCHEDULER_IMPL:
        impl = CUSTOM_SCHEDULER_IMPL[scheduler_name]["fn"]
        ref = _original_calculate_sigmas(model_sampling, "karras", steps)
        params = CUSTOM_SCHEDULER_IMPL[scheduler_name].get("default_params", {})
        return impl(model_sampling, ref, steps, params)
    return _original_calculate_sigmas(model_sampling, scheduler_name, steps)

cs.calculate_sigmas = calculate_sigmas_patched

# -------------------------------------------------------------------
# Node mappings (empty, unless you want to register nodes here)
# -------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
