# SABER: A Semantic-Aligned Brain Network Analysis Framework via Multi-scale Hypergraphs

Official implementation of **SABER**, a semantic-aligned brain network analysis framework for fMRI-based brain disease diagnosis.

SABER integrates LLM-derived semantic priors with multi-scale hypergraph neural networks. Instead of using language semantics only as auxiliary features, SABER injects semantic information from ROI-level representation learning to graph-level abstraction and decision-level classification.

## Highlights

- **ROI-level semantic injection**: anatomical and functional ROI descriptions are encoded and fused with functional connectivity features.
- **Multi-scale hypergraph modeling**: high-order multi-ROI interactions are modeled beyond pairwise graph edges.
- **Decision-level semantic alignment**: patient-specific textual embeddings directly guide graph-level classification.
- **Interpretable analysis**: the framework supports discriminative ROI and connection analysis for neurodevelopmental disorder diagnosis.

## Repository Structure

```text
.
--- llm-main-icme.py
--- baseline_icme.sh
--- configs/
--- data/
--- layers/
--- nets/
--- norm/
--- trainer.py
--- metrics.py
--- utils.py
--- requirements.txt
```

## Environment

```bash
conda create -n saber python=3.10 -y
conda activate saber
pip install -r requirements.txt
```

Install the DGL build that matches your CUDA/PyTorch environment if the default `dgl` package is not suitable.

## Data Preparation

Raw/preprocessed fMRI graph binaries are not included because public neuroimaging datasets have their own access and redistribution policies.

For the default ABIDE AAL116 experiment, prepare the DGL graph binary at:

```text
data/abide_full_AAL116/abide_full_AAL116.bin
```

The repository includes split files, atlas coordinates, and semantic prompt resources used by the training code. If you use another dataset, update `name2path` and `name2coor_path` in `data/BrainNet.py`.

## Training

```bash
bash baseline_icme.sh
```

Equivalent command:

```bash
python3 llm-main-icme.py   --gpu_id 0   --model BrainPromptMSHT   --dataset abide_full_AAL116   --config configs/abide_full_AAL116/TUs_graph_classification_BrainPromptMSHT_abide_full_AAL116_100k.json   --epochs 100   --batch_size 16   --init_lr 1e-4   --weight_decay 1e-4   --node_feat_transform pearson   --edge_ratio 0.2   --dropout 0.5   --lambda1 1.0
```

Outputs are written under `out/braindata_graph_classification/`.

## Paper

**SABER: A Semantic-Aligned Brain Network Analysis Framework via Multi-scale Hypergraphs**

Authors: Yidan Xu, Xiangmin Han, Rundong Xue, and Huihui Ye

Experiments in the paper are conducted on ABIDE and ADHD-200 using fMRI brain networks.

## Citation

```bibtex
@article{xu2026saber,
  title={SABER: A Semantic-Aligned Brain Network Analysis Framework via Multi-scale Hypergraphs},
  author={Xu, Yidan and Han, Xiangmin and Xue, Rundong and Ye, Huihui},
  journal={arXiv preprint arXiv:2607.01901},
  year={2026}
}
```

## Contact

For questions, please contact Yidan Xu: YidanXu2024@163.com.

## License

This project is released under the MIT License.
