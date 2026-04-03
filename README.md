
# Latent Semantic Surgery: Probing and Manipulating T2I Models

This repository introduces **Semantic Transplantation**, a research framework designed to probe and surgically manipulate the latent semantic space of Text-to-Image (T2I) diffusion models. By extending the *Semantic Surgery* framework (Xiong et al., NeurIPS 2025), we transition from concept erasure to a novel **Vector Injection** mechanism, enabling precise zero-shot attribute replacement.

## Core Contributions

### 1. From Erasure to Transplantation
While original methods focus on erasing concepts, we introduce a surgical injection formula designed to replace a source concept ($e_{src}$) with a target concept ($e_{new}$) directly within the CLIP embedding space:

$$e^{*} = e_{in} + \lambda \cdot M_{\alpha} \odot (e_{new} - e_{src})$$

**Legend:**
* $e^*$: The resulting "transplanted" latent embedding.
* $e_{in}$: The initial CLIP text embedding.
* $\lambda$: A scalar factor that scales the "injection force".
* $M_{\alpha}$: A spatial mask derived from token-wise similarity, thresholded by sensitivity $\alpha$.
* $\odot$: Denotes the Hadamard (element-wise) product.
* $(e_{new} - e_{src})$: The semantic direction vector for attribute replacement.

This allows for replacing subjects or contexts while preserving global context and structural fidelity.

### 2. SurgeryNet: Automated Parameter Prediction
To address the non-linearity of the hyperparameter landscape ($\lambda, \alpha$), we developed **SurgeryNet**, a Multi-Layer Perceptron (MLP) that automates the prediction of optimal surgical parameters directly from input text embeddings. SurgeryNet outperformed classical Random Forest baselines with a **12% error reduction** (MAE 0.0918 vs. 0.1043).

### 3. Latent Bias & Robustness Probing
A rigorous stress-test of Stable Diffusion's latent rigidity, identifying:
* **Shape Bias**: Dependency on geometric priors; morphologically similar swaps (e.g., $Dog \rightarrow Cat$) achieve high consistency ($IoU \approx 0.88$), while dissimilar ones (e.g., $Apple \rightarrow Daisy$) lead to structural hallucinations].
* **Attribute Entanglement**: "Visual leakage" where target concepts inherit source traits, such as a shark retaining an orange color when swapped from a goldfish.
* **Societal Bias**: Quantifying a **100% Gender Flip Rate** in specific occupational transformations (e.g., *Doctor* $\rightarrow$ *Nurse*).

## Repository Structure

* `src/`: Core implementation of the `StableDiffuser` class and evaluation metrics.
* `implementation_comparison/`: Side-by-side code comparison between the baseline (Xiong et al.) and our proposed Vector Injection framework.
* `results/`: Detailed ablation studies and context/subject swap outcomes.
* `Report.pdf`: Comprehensive technical paper detailing methodology, "Golden Score" metrics, and experimental findings.
* `semantic-transplant.ipynb`: Interactive demonstration and Grad-CAM attention visualization.

## Installation & Usage

```bash
pip install -r requirements.txt
```

Run the `semantic-transplant.ipynb` notebook to explore the automated "Surgery Autopilot" and visualize real-time semantic manipulations.

## Quantitative Highlights
* **SurgeryNet Performance**: 12% error reduction in parameter prediction compared to Random Forest.
* **Spatial Consistency**: Achieved $IoU \approx 0.88$ for morphologically similar subject swaps.
* **Golden Score**: Optimization based on a composite metric $S$ balancing CLIP scores and SSIM ($w_{c}=0.6, w_{s}=0.4$).

## Credits
Developed as part of the Advanced Machine Learning & Computer Vision curriculum (2025/2026). Based on the framework by Xiong et al. (2025).

**Authors**: Giulia Pietrangeli, Lorenzo Musso.

## License
MIT License
