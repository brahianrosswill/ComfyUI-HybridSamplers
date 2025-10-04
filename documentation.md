# Attention Multiplication Nodes: Comprehensive Documentation

This document provides a detailed explanation of the Attention Multiplication nodes found in `nodes_attention_multiply.py`. These nodes offer powerful, low-level control over the attention mechanisms in diffusion models, enabling fine-tuning of image and video generation.

## Core Concept: The `attention_multiply` Function

The underlying utility for most of these nodes is the `attention_multiply` function. It works by:
1.  **Cloning the Model**: It creates a safe, independent copy of the model to modify.
2.  **Accessing Parameters**: It reads the model's `state_dict`, which contains all learnable weights and biases.
3.  **Targeting Attention Layers**: It identifies the parameters of specific attention layers (e.g., `attn1` for self-attention, `attn2` for cross-attention).
4.  **Scaling Projections**: It multiplies the weights and biases of the Query (Q), Key (K), Value (V), and final Output (Out) projections by user-defined values.

This allows for the precise amplification or suppression of different components of the attention mechanism.

---

## 1. `UNetSelfAttentionMultiply`

This node manipulates the **self-attention** layers within the U-Net, which are responsible for the internal structure, coherence, and detail of a generated image. It targets `attn1` blocks.

| Parameter | Technical Role | Conceptual Effect | Practical Impact on Images |
| :--- | :--- | :--- | :--- |
| **`q`** | Scales Query vectors from each pixel. | "How intensely should I look for related pixels?" | **> 1.0**: Sharper details, more intricate textures. Can cause noise. <br> **< 1.0**: Softer, smoother image, less fine detail. |
| **`k`** | Scales Key vectors from each pixel. | "How loudly should I announce my own content?" | **> 1.0**: Higher contrast attention, sharper edges. Can create unnatural transitions. <br> **< 1.0**: More uniform attention, softer look. |
| **`v`** | Scales Value vectors from each pixel. | "How much of my attended content should be used?" | **> 1.0**: Amplifies the texture and features that are being attended to. Great for boosting detail. <br> **< 1.0**: Mutes or suppresses attended features, leading to a smoother image. |
| **`out`** | Scales the entire attention block's output. | "How much should self-attention influence the final result?" | **> 1.0**: More structurally coherent and organized image. <br> **< 1.0**: More dream-like or painterly effect; can lead to anatomical/structural errors if too low. |

---

## 2. `UNetCrossAttentionMultiply`

This node manipulates the **cross-attention** layers within the U-Net, which align the image with the text prompt. It targets `attn2` blocks, where the image queries the prompt's content.

| Parameter | Technical Role | Conceptual Effect | Practical Impact on Images |
| :--- | :--- | :--- | :--- |
| **`q`** | Scales Query vectors from the image. | "How intensely should the image seek meaning from the prompt?" | **> 1.0**: More literal prompt interpretation. Can cause visual confusion if overused. <br> **< 1.0**: More creative freedom, less constrained by the prompt. |
| **`k`** | Scales Key vectors from the text prompt. | "How distinct is the meaning of each word in the prompt?" | **> 1.0**: Better separation of concepts (e.g., "red ball," "blue cube"). Improves prompt clarity. <br> **< 1.0**: "Concept bleed," where attributes get mixed up. |
| **`v`** | Scales Value vectors from the text prompt. | "How much of the prompt's content should be injected into the image?" | **> 1.0**: Increases prompt strength and adherence. Makes the image look more like the prompt. <br> **< 1.0**: Weakens prompt influence, making it more subtle. |
| **`out`** | Scales the entire block's output. | "How much should the text prompt guide the overall generation?" | **> 1.0**: Similar to increasing CFG scale; stronger prompt guidance. <br> **< 1.0**: Weaker prompt guidance, more like unconditional generation. |

---

## 3. `CLIPAttentionMultiply`

This node operates on the **CLIP text encoder** itself. It modifies the self-attention between words in the prompt, changing how the prompt is *understood* before it is sent to the U-Net.

| Parameter | Technical Role | Conceptual Effect | Practical Impact on Prompt Interpretation |
| :--- | :--- | :--- | :--- |
| **`q`** | Scales Query vectors from each word token. | "How intensely should a word seek context from other words?" | **> 1.0**: Better understanding of grammar and long-range word relationships. <br> **< 1.0**: More literal, isolated interpretation of each word. |
| **`k`** | Scales Key vectors from each word token. | "How distinct is the semantic identity of each word?" | **> 1.0**: Sharpens the meaning of individual words before contextual blending. <br> **< 1.0**: Blurs semantic boundaries between words. |
| **`v`** | Scales Value vectors from each word token. | "How much meaning should be shared between words?" | **> 1.0**: Creates richer, more context-aware embeddings (e.g., "crown" becomes more "ornate"). <br> **< 1.0**: Reduces the amount of context shared between words. |
| **`out`** | Scales the entire block's output. | "How much should the model 'think' about the prompt's structure?" | **> 1.0**: More sophisticated, nuanced interpretation of the prompt. <br> **< 1.0**: More "vanilla" embeddings based on individual word meanings. |

---

## 4. `UNetTemporalAttentionMultiply`

This specialized node is for **video models**. It distinguishes between attention that happens *within* a frame (structural) and attention that happens *across* frames (temporal) to create consistency and motion.

| Parameter | Technical Role | Conceptual Effect | Practical Impact on Video |
| :--- | :--- | :--- | :--- |
| **`self_structural`** | Scales self-attention (`attn1`) within each frame. | Per-frame structural integrity. | **> 1.0**: More detailed and coherent individual frames. <br> **< 1.0**: Softer, more painterly frames. |
| **`self_temporal`** | Scales self-attention (`attn1`) across frames. | Frame-to-frame consistency. | **> 1.0**: Reduces flickering; creates stable, coherent animation. **Crucial for consistency.** <br> **< 1.0**: "Boiling" or morphing effect; can become chaotic. |
| **`cross_structural`** | Scales cross-attention (`attn2`) within each frame. | Per-frame prompt adherence. | **> 1.0**: Each frame follows the prompt more literally. <br> **< 1.0**: More creative freedom from the prompt in each frame. |
| **`cross_temporal`** | Scales cross-attention (`attn2`) across frames. | Prompt guidance for motion and transformation. | **> 1.0**: Better execution of prompts describing action ("a man running"). <br> **< 1.0**: Motion may not follow prompt's dynamic instructions accurately. |