# HybridSamplers for ComfyUI

HybridSamplers is a **ComfyUI custom node extension** that introduces new experimental **samplers** and **schedulers** designed to give more control and flexibility over the diffusion process.

With HybridSamplers, you can explore alternative numerical methods, dynamic scheduling strategies, and experimental noise shaping — useful for research, experimentation, or pushing creative generation beyond standard defaults.

---

## ✨ Features

### Custom Samplers

* **AdaptiveEuler** – Euler-based sampler with adaptive scaling.
* **DynamicLangevin** – Langevin dynamics with noise adaptation.
* **StochasticRungeKutta** – Runge–Kutta method with stochastic jitter.
* **TemporalSampling** – Time-aware blending between diffusion steps.
* **SpatialSampling** – Spatially perturbed sampling for added detail.
* **Quantized** – Rounds latents to discrete bins for a quantized look.
* **Anisotropic** – Applies anisotropic noise for directional detail.
* **MultiDimensional** – Hybrid Euler + Heun blending.

### Custom Schedulers

* **AdaptiveTime** – Time-decaying schedule.
* **DynamicSchedule** – Cosine/linear interpolation scaling.
* **VariableStep** – Adjustable step ranges.
* **ProgressiveDecay** – Interval-based decay factor.
* **AdaptiveExponential** – Growth + saturation scaling.
* **FractalTime** – Fractal-based scaling factor.
* **TemporalGradient** – Gradient-magnitude based schedule.
* **MemoryAware** – Retains partial state across steps.
* **MultiObjective** – Weighted averaging.
* **ResourceConstrained** – Efficiency-aware scaling.
* **DynamicWindow** – Adaptive vs fixed step scaling.
* **MultiAgent** – Multi-agent coordination scaling.

---

## 🔧 Installation

Clone into your `ComfyUI/custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/azazeal04/ComfyUI-HybridSamplers.git
```

Restart ComfyUI and the new **samplers** and **schedulers** will be available in dropdown menus.

---

## 🛠 Development

* Python ≥ 3.10
* Torch ≥ 2.0
* Requires ComfyUI installed and working.

---

## ⚠️ Notes

* These samplers/schedulers are **experimental** and may produce noisy or blurry results.
* Designed for testing new ideas in diffusion sampling, not guaranteed for production use.
* You can enable debug logs in the console to see when a custom sampler/scheduler is active.

---

## 📜 License

This project is licensed under the MIT License – see [LICENSE](LICENSE) for details.
