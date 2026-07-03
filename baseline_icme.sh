#!/usr/bin/env bash
set -euo pipefail

DATASET="abide_full_AAL116"
CONFIG="configs/${DATASET}/TUs_graph_classification_BrainPromptMSHT_${DATASET}_100k.json"

python3 llm-main-icme.py   --gpu_id 0   --model BrainPromptMSHT   --dataset "${DATASET}"   --config "${CONFIG}"   --epochs 100   --batch_size 16   --init_lr 1e-4   --weight_decay 1e-4   --node_feat_transform pearson   --edge_ratio 0.2   --dropout 0.5   --lambda1 1.0
