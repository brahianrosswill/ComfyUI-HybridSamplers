# HybridSamplers for ComfyUI

HybridSamplers is a **ComfyUI custom node extension** that introduces new experimental **samplers** and **schedulers** designed to give more control and flexibility over the diffusion process.

With HybridSamplers, you can explore alternative numerical methods, dynamic scheduling strategies, and experimental noise shaping — useful for research, experimentation, or pushing creative generation beyond standard defaults.

---

## ✨ Features

### Custom Samplers

* **AdaptiveEuler** – Euler-based sampler with adaptive scaling.
![AdaptiveEuler_00001](https://github.com/user-attachments/assets/10df36d8-3765-4c18-bde9-4bf730d3989a)

  
* **DynamicLangevin** – Langevin dynamics with noise adaptation.
![DynamicLangevin_00000](https://github.com/user-attachments/assets/ecd878f2-208b-476d-baca-34b92341a7ef)

  
* **StochasticRungeKutta** – Runge–Kutta method with stochastic jitter.
![StochasticRungeKutta_00000](https://github.com/user-attachments/assets/59f6c929-42d3-48bf-93c9-561cd4205108)

  
* **TemporalSampling** – Time-aware blending between diffusion steps.
![TemporalSampling_00000](https://github.com/user-attachments/assets/505b762f-7c0a-4e78-a664-fbf577f51e3a)

  
* **SpatialSampling** – Spatially perturbed sampling for added detail.
![SpatialSampling_00000](https://github.com/user-attachments/assets/bd839b18-ee56-4022-9cf7-db1fa5fee02e)

  
* **Quantized** – Rounds latents to discrete bins for a quantized look.
![Quantized_00000](https://github.com/user-attachments/assets/33e8830b-ea2e-45c8-a2ff-2491beb2a20b)

  
* **Anisotropic** – Applies anisotropic noise for directional detail.
![Anisotropic_00000](https://github.com/user-attachments/assets/a6e8bfc5-2820-4025-a701-dcc38bfead4c)

  
* **MultiDimensional** – Hybrid Euler + Heun blending.
![MultiDimensional_00000](https://github.com/user-attachments/assets/08070532-9a45-4b8b-9a57-16002a1e20d1)

Images generated at 15 steps with cfg 7.0 and seed 42

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
