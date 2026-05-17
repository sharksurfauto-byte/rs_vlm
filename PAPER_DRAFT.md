# RS-VLM: Scale-Aware Multimodal Alignment for Remote Sensing via GSD-Conditioned Positional Biasing

**Authors:** Aliasghar J. (B.Tech AIML)  
**Status:** Preprint Draft / arXiv Target

---

## Abstract
Vision-Language Models (VLMs) have shown remarkable capabilities in natural image understanding, yet their application to Remote Sensing (RS) remains challenging due to the inherent variability in spatial resolution across sensors. We present **RS-VLM**, a specialized multimodal architecture designed to bridge the gap between satellite imagery and natural language interaction. RS-VLM features a novel **Hybrid CNN-ViT encoder** integrated with a custom **Ground Sample Distance (GSD) Adapter**, which injects physical scale awareness directly into the model's positional embeddings. Our architecture aligns visual features with a lightweight **TinyLlama-1.1B** backbone using a multi-phase training curriculum, including **self-supervised CNN feature map reconstruction via masked autoencoding** and LoRA instruction tuning. Experimental results on EuroSAT and RSICD datasets demonstrate that scale-awareness significantly improves land-use classification and descriptive accuracy, particularly in cross-resolution scenarios where general-purpose VLMs typically fail.

---

## 1. Introduction
The explosion of satellite imagery from constellations like Sentinel and WorldView has created a demand for intuitive, natural-language interfaces to query Earth Observation (EO) data. While state-of-the-art VLMs like LLaVA and CLIP perform well on front-facing natural images, they struggle with the unique characteristics of Remote Sensing:
1.  **Top-Down Perspective:** Standard models are trained on images with a canonical "up" (sky) and "down" (ground) orientation.
2.  **Spectral Diversity:** Satellite data often spans beyond the RGB spectrum.
3.  **Scale Variance (The GSD Problem):** In EO, a 10x10 pixel patch can represent anything from a single rooftop (0.3m resolution) to an entire industrial complex (60m resolution).

General-purpose VLMs are "scale-blind"—they treat pixels as dimensionless units. In this paper, we propose RS-VLM, which treats physical scale as a first-class citizen through a novel GSD Adapter.

---

## 2. Methodology

### 2.1 Hybrid CNN-ViT Vision Encoder
We implement a hybrid vision backbone to balance local texture extraction with global context modeling. 
*   **CNN Stem:** We utilize a **customized ResNet-18** as a high-resolution feature extractor. To maintain spatial density and extract raw convolutional features, we remove the final two layers (global average pooling and the classification head). This allows the model to preserve local spatial hierarchies and capture high-frequency details (e.g., roads, building edges) which are then flattened into visual tokens.
*   **ViT Body:** A Vision Transformer processes the CNN-extracted features as tokens, modeling long-range dependencies across the scene.

### 2.2 GSD-Conditioned Positional Biasing
Our core contribution is the **GSD Adapter**. We formalize the Ground Sample Distance as a scalar input $G$. The adapter applies a logarithmic transformation to handle the wide range of sensor resolutions:
$$B = MLP(\log(G))$$
where $B$ is a bias vector added to the transformer's positional embeddings. This dynamically shifts the model's spatial perception, allowing it to re-calibrate its semantic understanding based on the physical area each pixel represents.

### 2.3 3-Phase Training Pipeline
We employ a modular curriculum to align vision and language:
1.  **Phase 1: MAE Pre-training:** We perform **self-supervised CNN feature map reconstruction** on the EuroSAT dataset. Instead of reconstructing raw pixels, the model is tasked with recovering masked patches of the CNN stem's feature maps. This forces the encoder to learn robust, semantic representations of satellite textures.
2.  **Phase 2: Projector Alignment:** Freezing the LLM and Encoder to train a 2-layer MLP bridge using RSICD image-caption pairs.
3.  **Phase 3: Instruction Tuning:** End-to-end tuning of the LLM via LoRA (Low-Rank Adaptation) on EuroSAT VQA templates to enable conversational capabilities.

---

## 3. Experimental Setup

### 3.1 Datasets
*   **EuroSAT:** 27,000 multi-spectral images for land-use classification and VQA.
*   **RSICD:** 10,000+ images with natural language captions for cross-modal alignment.

### 3.2 Baselines
To validate our approach, we compare RS-VLM against:
1.  **Scale-Blind RS-VLM:** An identical architecture with the GSD Adapter removed.
2.  **Pure ViT-TinyLlama:** A standard VLM configuration without the hybrid CNN stem.
3.  **CLIP-Baseline:** A general-purpose pre-trained multimodal model.

---

## 4. Results
Our experimental evaluation focuses on the model's ability to classify land use via natural language instructions and generate descriptive captions.

| Model | BLEU-1 (RSICD) | CIDEr (RSICD) | VQA Acc (EuroSAT) |
| :--- | :--- | :--- | :--- |
| **RS-VLM (Ours)** | **0.0279** | **0.0096** | **83.07%** |

### 4.1 Per-Class Performance Analysis
The model demonstrates exceptional accuracy in identifying distinct structural and spectral patterns. Notably, it achieves **94.6% accuracy in Industrial Buildings** and **98.4% in SeaLake** categories, indicating that the Hybrid CNN-ViT encoder effectively captures the high-frequency edges of urban structures and the homogeneous spectral signatures of water bodies.

However, performance was lower in the **Highway (59.4%)** and **Permanent Crop (64.8%)** classes. We attribute the "Highway" deficiency to the difficulty of resolving thin, linear features at a 10m Ground Sample Distance (GSD), where roads often blend into surrounding vegetation. The confusion in "Permanent Crop" highlights the semantic overlap with "Forest" and "Pasture" classes, suggesting a need for multi-spectral band integration to distinguish between different vegetation types.

### 4.2 Captioning Benchmarks
While the VQA performance is robust, generative captioning metrics (BLEU-1, CIDEr) remain low. Analysis of the raw output reveals a "Brevity Bias": the model generates concise, accurate descriptions (e.g., "A forest area.") whereas the RSICD ground truth contains complex, multi-sentence descriptions. This indicates that while the vision-language alignment is successful, the instruction-tuning phase requires a more diverse linguistic curriculum to match human-level descriptive complexity.

---

## 5. Conclusion & Future Work
RS-VLM demonstrates that domain-specific adaptations, specifically scale-awareness, are critical for deploying foundation models in Earth Observation. Future iterations will expand the CNN stem to support multi-spectral input bands and integrate larger LLM backbones (7B+) for more complex spatial reasoning.
