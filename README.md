# SABER: A Semantic-Aligned Brain Network Analysis Framework via Multi-scale Hypergraphs

This repository contains the official project page and implementation for:



## News

- Code and documentation will be released soon.
- The paper will be updated with the arXiv link once available.

## Overview

SABER is a semantic-aligned brain network analysis framework for brain disease diagnosis from functional magnetic resonance imaging (fMRI). The framework integrates large language model derived clinical semantics with multi-scale hypergraph neural networks, enabling semantic information to actively participate in decision-level brain network classification.

Unlike previous methods that mainly use semantic information as auxiliary features or supervision, SABER introduces semantic priors at multiple levels and aligns patient-specific textual embeddings with graph-level representations. This design improves classification robustness, stability, and interpretability, especially in small-sample neuroimaging scenarios.

## Key Ideas

- **ROI-level semantic injection**: anatomical and functional descriptions of brain regions are encoded and fused with connectivity-based node features.
- **Multi-scale hypergraph modeling**: multiple hypergraphs are constructed to capture high-order and multi-ROI interactions beyond pairwise connectivity.
- **Decision-level semantic alignment**: patient-specific semantic embeddings are selectively injected into graph representations through a graph-level alignment mechanism.
- **Interpretable brain network analysis**: the framework supports analysis of discriminative ROIs and functional connections.

## Framework

The SABER pipeline consists of three main stages:

1. **Multi-scale node-level brain network encoding**
   - Brain regions are represented using functional connectivity features.
   - ROI semantic embeddings are injected into node representations through global self-attention.

2. **Multi-scale hypergraph representation learning**
   - Hypergraphs at different scales model high-order dependencies among ROIs.
   - A gated fusion mechanism adaptively integrates cross-scale representations.

3. **Graph-level semantic alignment**
   - Patient-level textual semantics are aligned with graph representations.
   - Semantic information directly guides final disease classification while preserving structural brain network information.

## Datasets

Experiments are conducted on two public fMRI brain network datasets:

- **ABIDE I**: Autism Brain Imaging Data Exchange dataset for ASD diagnosis.
- **ADHD-200**: ADHD-200 dataset for Attention-Deficit/Hyperactivity Disorder diagnosis.

Both datasets are preprocessed using the DPARSF pipeline in the paper.





The BibTeX entry will be updated after the arXiv version is available.

## Contact

For questions or collaborations, please contact:

- Yidan Xu: YidanXu2024@163.com


## License

The license will be specified before the code release.
