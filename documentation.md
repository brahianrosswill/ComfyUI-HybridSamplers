# Attention Multiplication Nodes: A Guide to Creative Control

Welcome! This document is your guide to the Attention Multiplication nodes, a suite of tools designed to give you direct, creative control over the heart of a diffusion model: its attention mechanism. Think of these nodes as a mixing board for your model's imagination, allowing you to fine-tune how it sees, thinks, and creates.

## The Core Idea: What is Attention?

Imagine an artist painting a masterpiece. Their eyes don't focus on everything at once. They pay close attention to the texture of a brushstroke here, the reflection in an eye there, and the overall composition everywhere. This selective focus is what attention mechanisms do for AI. They help the model decide which parts of an image (or a text prompt) are most important and how they relate to each other.

Our nodes let you adjust the "volume" of different parts of this attention process. By turning these knobs, you can guide the model's focus, leading to dramatic changes in style, coherence, and prompt adherence.

### How It Works: The `attention_multiply` Function

Under the hood, these nodes use a function called `attention_multiply`. You don't need to be a programmer to use it, but here’s the simple analogy:

1.  **It safely duplicates the artist's canvas**: It clones the AI model so your original is never harmed.
2.  **It finds the artist's toolkit**: It locates the specific parts of the model responsible for attention—the Query, Key, Value, and Output projections.
3.  **It adjusts the tools**: It amplifies or quiets down these components based on your input. For example, it can make the artist's "querying" eye more intense or the "value" of what they see more impactful.

This gives you a powerful way to experiment and discover new visual styles without complex training.

---

## 1. `UNetSelfAttentionMultiply`

**Analogy: The Internal Critic**

Think of self-attention as the artist's internal critic, responsible for making sure the painting is coherent and well-structured. It's the process of stepping back and asking: "Do the shadows match the light source? Does this character's anatomy make sense? Do the textures feel consistent?" This node lets you tune how strict or imaginative that internal critic is.

It manipulates the **self-attention** layers within the U-Net (`attn1` blocks), which are responsible for the internal structure, coherence, and detail of a generated image.

| Parameter | Conceptual Effect | Practical Impact & Example Use Cases |
| :--- | :--- | :--- |
| **`q` (Query)** | "How intensely should I look for related pixels?" | **> 1.0**: "I want hyper-realistic skin pores" or "Make the fractal patterns incredibly intricate." Use this to push fine details to their limit, but be careful of adding visual static or noise. <br> **< 1.0**: "Give me a soft, dreamy portrait" or "I want a smooth, cel-shaded look." This softens the focus on tiny details, creating a more blended, gentle appearance. |
| **`k` (Key)** | "How loudly should I announce my own content?" | **> 1.0**: "Make the difference between 'brick' and 'mortar' textures extremely clear." This increases the contrast between different elements, making edges and material definitions pop. Can look artificial if pushed too far. <br> **< 1.0**: "Blend the foreground and background with a gentle, foggy haze." This reduces the sharp distinctions between elements, creating a more unified and harmonious composition. |
| **`v` (Value)** | "How much of my attended content should be used?" | **> 1.0**: "Once you've identified a 'metallic' texture, make it *really* shiny and reflective." This amplifies the features the model decides to focus on. If it finds a detail, it will emphasize it. <br> **< 1.0**: "I like the overall composition, but tone down the busy patterns in the wallpaper." This mutes the attended features, making them less prominent without removing them entirely. |
| **`out` (Output)** | "How much should self-attention influence the final result?" | **> 1.0**: "The overall structure is paramount. A person must have two arms and two legs." This boosts the final say of the self-attention mechanism, enforcing strong structural integrity. <br> **< 1.0**: "Let go of realism; I want a surreal, melting-clocks effect." This reduces the influence of the structural check, allowing for more abstract, painterly, or dream-like results. Too low, and your image might fall apart into a mess. |

---

## 2. `UNetCrossAttentionMultiply`

**Analogy: A Game of Charades**

Cross-attention is like a game of charades or Pictionary between the image and the text prompt. The image is the player trying to guess the word, and the prompt is the one giving the clues. The image constantly looks at the prompt (`Query`) to understand what it's supposed to become. This node lets you direct that game, making the players better communicators.

It targets the **cross-attention** layers (`attn2` blocks), where the image queries the prompt's content to ensure it's following instructions.

| Parameter | Conceptual Effect | Practical Impact & Example Use Cases |
| :--- | :--- | :--- |
| **`q` (Query)** | "How intensely should the image seek meaning from the prompt?" | **> 1.0**: "Stick to the prompt, no matter what." This makes the image generation process much more literal. Useful for complex prompts where you need every detail to be present. Overdo it, and you might get a chaotic image that tries to include everything at once. <br> **< 1.0**: "The prompt is just a suggestion." This gives the model more creative freedom to wander away from the text. Great for discovering unexpected compositions and styles. |
| **`k` (Key)** | "How distinct is the meaning of each word in the prompt?" | **> 1.0**: Solves "concept bleed." If your prompt is "a red cube and a blue sphere," and you're getting a "purple cube," increasing `k` helps the model distinguish "red" from "blue" more clearly. It sharpens the definition of each clue. <br> **< 1.0**: "Let the ideas mix and mingle." This can be used for creative effect, intentionally blending concepts to create surreal, hybrid objects and scenes. |
| **`v` (Value)** | "How much of the prompt's content should be injected into the image?" | **> 1.0**: "The prompt says 'a cyber-punk city,' so make it look *aggressively* cyberpunk." This boosts the visual strength of the concepts found in the prompt. It's like turning up the volume on the prompt's stylistic and subject-matter instructions. <br> **< 1.0**: "I want a portrait of a person, with just a *hint* of a fantasy setting." This makes the prompt's influence more subtle, like a whisper rather than a shout. |
| **`out` (Output)** | "How much should the text prompt guide the overall generation?" | **> 1.0**: "The prompt is the boss." This is very similar to increasing the CFG scale, forcing the model to pay close attention to the prompt across the board. It's a great way to increase overall prompt adherence. <br> **< 1.0**: "Generate an image that feels *unconditionally* creative." This is like lowering the CFG scale, making the final image less guided by the text and more by the model's own internal creativity. |

---

## 3. `CLIPAttentionMultiply`

**Analogy: The Language Scholar**

Before an artist can paint "a king holding a golden scepter," they must first understand what that *means*. This node adjusts the **CLIP text encoder**, which is the language scholar inside the AI. It modifies how the words in your prompt talk to *each other* to figure out the overall meaning. Is "golden" modifying "king" or "scepter"? How important is the relationship between "holding" and the two nouns? This is where you can fine-tune that internal dialogue.

It modifies the self-attention between words in the prompt, changing how the prompt is *understood* before it is sent to the U-Net.

| Parameter | Conceptual Effect | Practical Impact & Example Use Cases |
| :--- | :--- | :--- |
| **`q` (Query)** | "How intensely should a word seek context from other words?" | **> 1.0**: "Understand the grammar deeply." For a prompt like "a futuristic car on a road that is wet from rain," this helps the model connect "wet" to "road" and not "car." It's great for long, descriptive prompts. <br> **< 1.0**: "Treat each word more literally and independently." This can be useful for style-focused prompts like "masterpiece, 8k, hyperdetailed" where the relationship between words is less important than their individual impact. |
| **`k` (Key)** | "How distinct is the semantic identity of each word?" | **> 1.0**: "A 'king' is a king, and a 'scepter' is a scepter. Don't confuse them." This sharpens the core meaning of each word, making them more distinct before they get blended with context. <br> **< 1.0**: "Let the word meanings be a bit more fluid." This can lead to more poetic or abstract interpretations, where the boundaries between concepts are intentionally blurred. |
| **`v` (Value)** | "How much meaning should be shared between words?" | **> 1.0**: "When 'golden' is next to 'scepter,' make the concept of 'scepter' more regal and ornate." This encourages words to enrich their neighbors, creating more nuanced and context-aware concepts before they are passed to the image generator. <br> **< 1.0**: "Just give me the basic, dictionary definition of each word." This reduces the contextual influence, leading to a more straightforward, if less sophisticated, interpretation. |
| **`out` (Output)** | "How much should the model 'think' about the prompt's structure?" | **> 1.0**: "The final interpretation should be highly sophisticated." This boosts the power of the whole 'language scholar' process, leading to a more nuanced understanding of complex sentences. <br> **< 1.0**: "A simple interpretation is fine." This results in a more 'vanilla' understanding of the prompt, relying on the most basic meanings of the words. |

---

## 4. `UNetTemporalAttentionMultiply`

**Analogy: The Animator's Desk**

This specialized node is for **video models**. Think of it as an animator's desk, with two key responsibilities:
1.  **Drawing a single, beautiful frame (Structural Attention)**: This is about making sure each individual picture is well-composed, detailed, and internally coherent.
2.  **Flipping through the frames to create smooth motion (Temporal Attention)**: This is about making sure the character looks the same from one frame to the next, that the background is stable, and that movement feels natural.

This node gives you separate controls for the self-attention (the image's internal consistency) and cross-attention (the image's adherence to the prompt) for both of these responsibilities.

| Parameter | Conceptual Effect | Practical Impact & Example Use Cases |
| :--- | :--- | :--- |
| **`self_structural`** | "How detailed should each individual frame be?" | **> 1.0**: "I want every frame to be a cinematic masterpiece." This increases the internal detail and coherence of each frame, independent of others. <br> **< 1.0**: "Each frame should have a soft, impressionistic style." This reduces the fine detail within each frame, which can be a stylistic choice. |
| **`self_temporal`** | "How much should a frame look like the previous one?" | **> 1.0**: **This is the key to consistency.** "Stop the flickering!" or "Make the character's face stay the same." Turn this up to create stable, coherent animations where objects and people maintain their identity over time. <br> **< 1.0**: "I want a boiling, morphing, dream-like effect." Lowering this value makes elements change and shift from one frame to the next, which can be great for magical effects or chaotic scenes, but disastrous for realistic animation. |
| **`cross_structural`** | "How closely should this single frame follow the prompt?" | **> 1.0**: "The prompt says 'a detailed portrait,' so make sure every frame is a perfect portrait." This enforces strict, per-frame prompt adherence. <br> **< 1.0**: "Let the frames have some creative freedom from the prompt." This allows for more variation in how each individual frame interprets the text. |
| **`cross_temporal`** | "How well should the *motion* follow the prompt?" | **> 1.0**: "The prompt is 'a car driving down the street,' so make the car *move* correctly." This is crucial for accurately animating actions described in the prompt, like "running," "blooming," or "transforming." <br> **< 1.0**: "The prompt describes an action, but the animation can be more interpretive or subtle." The motion in the video may not follow the prompt's dynamic instructions as literally. |

---

## 5. General Tips & Tricks

Here are some best practices to get the most out of these nodes.

*   **Start Small**: These parameters are powerful. A value of 1.5 is a *huge* increase, and a value of 0.5 is a *huge* decrease. Start with small adjustments (e.g., 1.1 or 0.9) and observe the effect before making drastic changes.
*   **Isolate and Test**: When first learning, use only one attention node at a time. This helps you develop an intuition for what each parameter does without the confounding influence of others.
*   **Combine for Complex Effects**: Once you are comfortable, you can achieve sophisticated results by chaining nodes.
    *   **Example 1: Fixing Prompt Grammar & Bleed**: Use `CLIPAttentionMultiply` with a slightly higher `q` (e.g., 1.1) to improve grammar, then feed that into a `UNetCrossAttentionMultiply` with a higher `k` (e.g., 1.1) to prevent concept bleed.
    *   **Example 2: Detailed and Coherent Images**: Use `UNetSelfAttentionMultiply` with a slightly higher `v` and `out` (e.g., 1.05) to boost details and structure, then use `UNetCrossAttentionMultiply` to ensure it still adheres to the prompt.
*   **Troubleshooting: When Things Go Wrong**:
    *   **Too much noise/static?** You've likely pushed the `q` parameter in `UNetSelfAttentionMultiply` too high.
    *   **Image looks chaotic or "fried"?** Your values are probably too extreme. Pull everything back closer to 1.0. Often, the `out` or `v` parameters in any of the nodes can cause this if set too high.
    *   **Anatomy is falling apart?** Your `out` value in `UNetSelfAttentionMultiply` might be too low.
    *   **Video is flickering uncontrollably?** Your `self_temporal` value in `UNetTemporalAttentionMultiply` is too low. This is the most important parameter for video consistency.
*   **The `out` Parameter is a Big Hammer**: The `out` parameter on each node is a global multiplier for that entire attention block. If you want to make a subtle change, adjust `q`, `k`, or `v`. If you want to make a big, overarching change to that attention mechanism's influence, adjust `out`.