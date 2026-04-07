# HIBI-Net: A Lightweight Hybrid Framework with Multi-Scale Context Aggregation and Boundary Regularization for Polyp Segmentation.

This repository contains the official PyTorch implementation of **HIBI-Net**, a hybrid Transformer–CNN architecture designed for high-precision polyp segmentation.  
HIBI-Net integrates **multi-scale kernel interactions**, **enhanced spatial-domain fusion**, and **explicit boundary supervision** to achieve accurate and anatomically coherent segmentation across diverse colonoscopy datasets.

---

##  Overview

HIBI-Net addresses three long-standing challenges in automated polyp segmentation:

- **Large variation in polyp scale and morphology**
- **Weak or ambiguous boundaries**
- **Need for both global semantic context and fine spatial detail**

To tackle these issues, BiSK-Net introduces three lightweight yet effective components:

1. **Multi-scale Kernel Interaction (MKI) Bottleneck**  
   Aggregates spatial information using parallel depthwise convolutions and feature-adaptive gating.

2. **Enhanced Spatial-Domain Interaction (eSDI) Module**  
   Improves semantic–spatial alignment using SE-based recalibration.

3. **Boundary-Aware Supervision**  
   Encourages sharper and more structurally consistent predictions using Sobel-derived boundary maps.

