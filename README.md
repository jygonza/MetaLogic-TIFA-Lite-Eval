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

- **Commutative–Horizontal:  **
  - A: `a red cube to the left of a blue sphere on a wooden table`  
  - B: `a blue sphere to the right of a red cube on a wooden table`  

- **DeMorgan–Conjunctive:  **
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

Outputs are saved as:
``text
Images/<logical_law>/<category_id>/<semantic_dimension>/<pair_id>_A.png
Images/<logical_law>/<category_id>/<semantic_dimension>/<pair_id>_B.png
``

### 3. CLIPScore + VQA Evaluation

The metrics we use to evaluate for robustness are **CLIPScore** and a **TIFA-lite VQA evaluation**. The `MetaLogic-Image-Eval.ipynb` notebook implements these metrics, generating CLIPScores for each image prompt pair, and a VQA pipeline.

We implement a simplified TIFA-style pipeline:

1. Use an LLM (e.g., GPT-4.1 mini) to generate a small set of question–answer pairs per prompt.
2. Use BLIP/BLIP-2 VQA to answer those questions given the generated image.
3. Compare predicted answers vs. gold answers and compute VQA accuracy per image.

We treat this accuracy as a faithfulness score in [0,1].

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

[category_metrics.csv](Plotted_Results/category_metrics.csv)

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
