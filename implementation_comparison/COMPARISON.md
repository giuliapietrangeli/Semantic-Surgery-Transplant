# Architecture Comparison: Semantic Surgery vs. Original Implementation

This directory provides a direct code-level comparison between the original method (based on Concept Erasure) and our proposed **Semantic Surgery** framework.

## Files

*   **`utils_original.py`**: The baseline implementation, focusing on Concept Erasure via global subtraction.
*   **`utils_proposed.py`**: Our modified implementation, introducing Vector Injection ("Semantic Transplant") and Token-Wise Precision.

## Key Technical Modifications

Our modifications primarily target the `StableDiffuser` class and introduce new similarity metrics. Below is a breakdown of the core discrepancies.

### 1. Fine-Grained Token Similarity (Attention Mechanism)

The original implementation relied on global cosine similarity, which often led to wide-area erasures affecting context. We implemented a token-wise approach.

*   **Original (`utils_original.py`, `compute_similarity`)**:
    *   Uses simple `F.cosine_similarity` averaged over the sequence.
    *   Result: Single scalar similarity score.

*   **Proposed (`utils_proposed.py`, `compute_token_similarity`)**:
    *   Computes an interaction matrix between *every token* in the prompt and the target concept.
    *   Uses Max-Pooling (`sim_matrix.max(dim=-1)`) to find the highest correlation for each specific word.
    *   **Impact**: Creates a precise spatial mask (`M_alpha`), allowing the model to target specific words (e.g., "dog") while leaving "sits on a sofa" untouched.

### 2. Semantic Transplant (Vector Injection)

The most significant architectural change is the ability to *replace* concepts rather than just deleting them.

*   **Original (`get_multi_erased_embedding`)**:
    *   Only supports subtraction: $c^* = c_{in} - \lambda M_\alpha (v_{target} - v_{neutral})$
    
*   **Proposed (`get_multi_erased_embedding`)**:
    *   Introduces `text_replace` argument.
    *   **Vector Injection Logic**:
        ```python
        if text_replace:
            embedding_output = embedding_in + alpha_mask * (embedding_replace - embedding_erase)
        ```
    *   **Impact**: Enables "Semantic Surgery" (e.g., turning a "dog" into a "cat" while preserving posture), rather than just "Amnesia".

### 3. Hyperparameter Mapping

We re-mapped the abstract concepts of "Sensitivity" and "Force" to the internal mathematical parameters for better control.

*   **Proposed Logic**:
    *   **Sensitivity** maps to `beta` (sigmoid bias), controlling how strict the attention mask is.
    *   **Force** maps to `lambda` (scalar multiplier), controlling the intensity of the vector injection.

### 4. Interface Updates

The `__call__` method was updated to propagate these capabilities to the high-level API.

*   **Added**: `replace_with` argument in the main inference pipeline.
*   **Feedback Loop**: The "Visual Detection Feedback" (LCP) loop was updated to support replacement reinforcement, ensuring that if a transplant is too weak, the second pass amplifies the injection, not just the erasure.
