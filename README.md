# MetaLogic Robustness Evaluation of Stable Diffusion

This repository contains the code and notebooks for our Trustworthy AI Final Project, where we evaluate the **logical and semantic robustness** of a text-to-image (T2I) diffusion model.

We focus on **Stable Diffusion v1.5** and evaluate how robust it is to **logically equivalent prompt perturbations** using a MetaLogic-inspired framework and a TIFA-lite style VQA pipeline.

---

## Project Overview

We design a set of prompt pairs based on:

- **5 Logical Laws**
  - Commutative: $(A \land B \equiv B \land A\)$
  - Associative: $((A \land B) \land C \equiv A \land (B \land C)\)$  
  - Distributive: $(A \land (B \lor C) \equiv (A \land B) \lor (A \land C)\)$  
  - Complement (Negation): $(A\) vs. \(\neg A\)$  
  - DeMorgan: $(\neg(A \land B) \equiv \neg A \lor \neg B\$

- **4 Semantic Dimensions**
  - **Conjunctive** – object presence / omission  
  - **Horizontal** – left/right spatial relations  
  - **Vertical** – above/below spatial relations  
  - **Attributes** – color, size, category binding  

This yields **20 MetaLogic categories** = 5 logical laws × 4 semantic dimensions.

For each category we generate multiple **prompt pairs (A, B)** that are *logically equivalent* in intent.

We then:

1. Generate images with **Stable Diffusion v1.5** (and optionally SDXL).  
2. Compute **CLIPScore** between prompts and images.  
3. Use an **LLM + BLIP VQA pipeline** to estimate **faithfulness** and **logical consistency** (TIFA-lite).  
4. Aggregate results by **category, logical law, semantic dimension**, and visualize robustness patterns.

---

## Method Summary

### 1. Prompt Generation (MetaLogic Templates)

We generate templated prompts over simple objects (cat, dog, banana, etc.) with controlled relations, such as:

- Commutative–Horizontal:  
  - A: `a red cube to the left of a blue sphere on a wooden table`  
  - B: `a blue sphere to the right of a red cube on a wooden table`  

- DeMorgan–Conjunctive:  
  - A: `a scene where it is not the case that both a red cube and a blue sphere are present`  
  - B: `a scene with either no red cube or no blue sphere present`  

The notebook `generate_metalogic_prompts.ipynb` creates `prompts_metalogic.csv` with:

- `pair_id`
- `category_id`
- `logical_law`
- `semantic_dimension`
- `prompt_A`
- `prompt_B`

### 2. Image Generation (Stable Diffusion)

We use **Stable Diffusion v1.5** via Hugging Face `diffusers`:

- Resolution: `512 × 512`  
- Steps: e.g., 30  
- Guidance scale: e.g., 7.5  
- Scheduler: DPMSolver++ (2M)  
- Fixed random seed per prompt pair for reproducibility

We also optionally evaluate using **Stable Diffusion XL (SDXL)** for higher-resolution generations when an A100 GPU is available:

- Resolution: `1024 × 1024`
- Steps: e.g., 30
- Guidance scale: 7.0–8.0
- Scheduler: DPMSolver++ (2M)
- Mixed precision: `fp16`
- Safety checker disabled for consistency across prompts


Outputs are saved as:
``text
Images/<logical_law>/<category_id>/<semantic_dimension>/<pair_id>_A.png
Images/<logical_law>/<category_id>/<semantic_dimension>/<pair_id>_B.png
``

### 3. CLIPScore + VQA Evaluation

Our evaluation combines two complementary metrics:

1. **CLIPScore** — measures how well an image matches its own prompt  
2. **TIFA-lite VQA** — checks whether the *visual facts* in the image remain consistent across logically equivalent prompt pairs

Together, these evaluate both **global semantic fidelity** (CLIP) and **fine-grained factual grounding** (VQA).

---

### **3.1 CLIPScore (Global Semantic Alignment)**

For each generated image, we compute a CLIP-based similarity between the image and its corresponding prompt.  
Scores are scaled to `[0, 100]`.

For each logically equivalent pair `(A, B)`:

- `clip_A = CLIP(image_A, prompt_A)`
- `clip_B = CLIP(image_B, prompt_B)`
- `clip_diff = |clip_A − clip_B|`
- `clip_stability = 1 − clip_diff`  
  *(1 = perfectly invariant under the logical perturbation)*

This measures whether the model preserves the **overall meaning** of logically equivalent prompts.

---

### **3.2 TIFA-lite VQA (Fine-Grained Semantic Checking)**

CLIP cannot detect fine details such as incorrect colors, object omissions, or violations of spatial relations.  
To evaluate these, we implement a lightweight version of **TIFA (Text-to-Image Faithfulness Assessment)**.

TIFA is a benchmark that evaluates how faithfully a generated image reflects the *specific factual content* of a prompt using VQA (Visual Question Answering).  
We implement a simplified variant (“TIFA-lite”) appropriate for our datase

### 4. Results

The `MetaLogic-Faithfulness-Eval.ipynb` notebook takes the raw VQA outputs and computes **category-level metrics**, focusing on analyzing the raw results over the 20 individual perturbation categories, the 5 logical laws, and the 4 semantic dimensions.

For each image, it computes **VQA accuracy** as:
* faithfulness = fraction of questions answered correctly

For each prompt pair (A, B), it computes:
* faith_A: VQA accuracy on image from prompt A
* faith_B: VQA accuracy on image from prompt B
* faith_mean = (faith_A + faith_B) / 2
* faith_diff = |faith_A − faith_B|
* faith_stability = 1 − faith_diff (1 = perfectly invariant, 0 = maximally unstable)

The notebook also produces:
* Bar plots of stability by category (20 bars)
* Bar plots by logical law (5 bars)
* Bar plots by semantic dimension (4 bars)
* Heatmaps (logical law × semantic dimension) of mean stability and mean faithfulness

[category_faithfulness_bar.png](Plotted_Results/category_faithfulness_bar.png)

[category_stability_bar.png](Plotted_Results/category_stability_bar.png)

[logical_law_stability_bar.png](Plotted_Results/logical_law_stability_bar.png)

[semantic_dim_faithfulness_bar.png](Plotted_Results/semantic_dim_faithfulness_bar.png)

[heatmap_stability.png](Plotted_Results/heatmap_stability.png)

[heatmap_faithfulness.png](Plotted_Results/heatmap_faithfulness.png)

**Qualitative examples of the worst-performing categories** (A/B images + prompts)
[Results Report PDF](Plotted_Results/MetaLogic_TIFA_Report.pdf)
