# Semantic Surgery: Stress-Testing Text-to-Image Models

This repository contains the implementation of **Semantic Surgery**, a framework for probing and manipulating the latent semantic space of Text-to-Image models (specifically Stable Diffusion). This project explores methods like **Vector Injection** and **Concept Erasure** to analyze model sensitivity, bias, and robustness.

## Project Structure

This project focuses on the following core components:

*   **`src/`**: The main source code directory containing the implementation of the Semantic Surgery framework.
    *   `utils.py`: Defines the `StableDiffuser` class, which handles the core logic for semantic vector manipulation (injection, erasure) and attention masking.
    *   `SS_inference.py`: Provides utilities for batch inference and image generation pipelines.
    *   `evaluation.py` & `detection_*.py`: Scripts for evaluating model performance and detecting objects/concepts.
*   **`semantic-translplant.ipynb`**: The primary Jupyter Notebook for interacting with the project. It demonstrates the "Semantic Transplant" methodology, allowing users to perform experiments, visualize attention maps (Grad-CAM), and explore the "Surgery Autopilot".
*   **`Report.pdf`**: A comprehensive technical report detailing the theoretical background, methodology (Semantic Integrity, Structural Fidelity), experimental results, and in-depth analysis of phenomena like "Topological Isomorphism" and "Contextual Leakage".
*   **`requirements.txt`**: A list of Python dependencies required to run the project.

## Implementation Comparison / Transparency

To provide full transparency on our contributions, we have included a **Comparison Module** in the `implementation_comparison/` directory. This folder allows for a direct code-level comparison between the original "Concept Erasure" method and our proposed "Semantic Surgery" framework.

*   **`utils_original.py`**: The baseline implementation (Reference).
*   **`utils_proposed.py`**: Our modified implementation featuring **Vector Injection** and **Token-Wise Precision**.
*   **`COMPARISON.md`**: A detailed documentation of the architectural changes and novelties introduced.

## Getting Started

### Prerequisites

Ensure you have Python installed. You can install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Usage

1.  **Interactive Exploration**: Open and run `semantic-translplant.ipynb` to step through the "Semantic Surgery" process, from environment setup to live demonstrations. This notebook serves as the main entry point for understanding and verifying the method.
2.  **In-Depth Analysis**: Refer to `Report.pdf` for a detailed explanation of the algorithms, experimental setup, and findings.

## Notes

*   This project uses `diffusers` and `transformers` libraries for stable diffusion and CLIP models.
*   Hardware acceleration (CUDA/MPS) is recommended for inference.

---
*University Project - Advanced Machine Learning*
