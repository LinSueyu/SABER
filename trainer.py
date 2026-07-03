"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import csv
import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold,datasets
from metrics import accuracy_TU as accuracy, precision, recall, f1, roc_auc, accuracy_all_classes
from captum.attr import IntegratedGradients
from functools import partial

"""
    For GCNs
"""
def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels, batch_llms) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_llms = batch_llms.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        if model.name in ["PRGNN", "LINet"]:
            batch_scores, score1, score2 = model.forward(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels, score1, score2)
        #elif model.name in ['BrainPromptG', 'BrainPromptC', 'MMbrainG']:
        elif model.name in ['MMbrainG']:
            z1, z2, batch_scores = model.forward(batch_graphs, batch_x, batch_e,batch_llms)
            loss = model.loss(z1, z2, batch_scores, batch_labels)
        else:
            batch_scores = model.forward(batch_graphs, batch_x, batch_e,batch_llms)
            #batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels,batch_llms) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_llms = batch_llms.to(device)

            batch_labels = batch_labels.to(device)
            if model.name in ["PRGNN", "LINet"]:
                batch_scores, score1, score2 = model.forward(batch_graphs, batch_x, batch_e)
                loss = model.loss(batch_scores, batch_labels, score1, score2)
            #elif model.name in ['BrainPromptG', 'BrainPromptC', 'MMbrainG']:
            elif model.name in ['MMbrainG']:
                z1, z2, batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_llms)
                loss = model.loss(z1, z2, batch_scores, batch_labels)
            else:
                #batch_scores = model.forward(batch_graphs, batch_x, batch_e)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_llms)
                loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    return epoch_test_loss, epoch_test_acc


# def evaluate_network_all_metric(model, device, data_loader, epoch=0, path=''):
#     model.eval()
#     epoch_test_loss = 0
#     epoch_test_acc = 0
#     nb_data = 0
#     labels = []

#     def saliency_forward(x, e, llm, g):
#         return model.forward(g, x, e, llm)

#     with torch.no_grad():
#         y_pred = []
#         maps = []
#         for iter, (batch_graphs, batch_labels, batch_llms) in enumerate(data_loader):
#             batch_graphs = batch_graphs.to(device)
#             batch_x = batch_graphs.ndata['feat'].to(device)
#             batch_e = batch_graphs.edata['feat'].to(device)
#             batch_llms = batch_llms.to(device)

#             batch_labels = batch_labels.to(device)
#             if model.name in ["PRGNN", "LINet"]:
#                 batch_scores, score1, score2 = model.forward(batch_graphs, batch_x, batch_e)
#                 loss = model.loss(batch_scores, batch_labels, score1, score2)
#             #elif model.name in ['BrainPromptG', 'BrainPromptC', 'MMbrainG']:
#             elif model.name in ['MMbrainG']:
#                 z1, z2, batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_llms)
#                 loss = model.loss(z1, z2, batch_scores, batch_labels)
#             else:
#                 #batch_scores = model.forward(batch_graphs, batch_x, batch_e)
#                 batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_llms)
#                 loss = model.loss(batch_scores, batch_labels)

#                 # generate saliency maps
#                 if path:
#                     ig = IntegratedGradients(partial(saliency_forward, e=batch_e, llm=batch_llms, g=batch_graphs))
#                     # saliency = ig.attribute(batch_x, n_steps=1, target=0)
#                     # maps.append(saliency.reshape(-1, 116, 116))
#                     saliency = ig.attribute(batch_x, n_steps=8, target=0)

#                     # saliency: (B*116, F) or (B, 116, F)
#                     if saliency.dim() == 2:
#                         saliency = saliency.view(-1, 116, saliency.size(-1))

#                     # ROI-level importance: mean over feature dim
#                     roi_saliency = saliency.abs().mean(dim=-1)   # (B, 116)

#                     maps.append(roi_saliency)


#             labels.append(batch_labels.detach().cpu())
#             y_pred.append(batch_scores.detach().cpu())

#             epoch_test_loss += loss.detach().item()
#             epoch_test_acc += accuracy(batch_scores, batch_labels)
#             nb_data += batch_labels.size(0)

#         epoch_test_loss /= (iter + 1)
#         epoch_test_acc /= nb_data

#         y_true = torch.cat(labels, dim=0).numpy()
#         y_pred = torch.cat(y_pred, dim=0).numpy()

#         if len(np.unique(y_true)) == 2:
#             test_precision = precision(y_pred, y_true)
#             test_recall = recall(y_pred, y_true)
#             test_f1 = f1(y_pred, y_true)
#             test_roc_auc = roc_auc(y_pred, y_true)
#         else:
#             test_precision = epoch_test_acc
#             test_recall = epoch_test_acc
#             test_f1 = epoch_test_acc
#             test_roc_auc = epoch_test_acc

#         all_acc = accuracy_all_classes(y_pred, y_true)

#         # if path:
#         #     maps = torch.cat(maps, dim=0)
#         #     torch.save(maps, path)
#         if path:
#             maps = torch.cat(maps, dim=0)   # (N_samples, 116)

#             # average over subjects
#             roi_score = maps.mean(dim=0)    # (116,)

#             # Top-10 ROIs
#             topk = torch.topk(roi_score, k=10)
#             topk_indices = topk.indices.cpu().numpy()
#             topk_scores = topk.values.cpu().numpy()

#             # save results
#             np.save(path.replace('.pt', '_roi_score.npy'), roi_score.cpu().numpy())

#             with open(path.replace('.pt', '_top10_roi.txt'), 'w') as f:
#                 for i, (idx, score) in enumerate(zip(topk_indices, topk_scores)):
#                     f.write(f'{i+1}: ROI {idx}, Score = {score:.6f}\n')

#             print('Top-10 ROI indices:', topk_indices)
#             print('Top-10 ROI scores:', topk_scores)


#     return epoch_test_loss, epoch_test_acc, test_precision, test_recall, test_f1, test_roc_auc, all_acc

# def evaluate_network_all_metric(model, device, data_loader, epoch=0):
#     """
#     Evaluate network metrics and compute ROI-level saliency scores.
#     Top-10 ROIs will be saved in brain.txt
#     """
#     model.eval()
#     epoch_test_loss = 0
#     epoch_test_acc = 0
#     nb_data = 0
#     labels = []

#     def saliency_forward(x, e, llm, g):
#         return model.forward(g, x, e, llm)

#     with torch.no_grad():
#         y_pred = []
#         maps = []
#         for iter, (batch_graphs, batch_labels, batch_llms) in enumerate(data_loader):
#             batch_graphs = batch_graphs.to(device)
#             batch_x = batch_graphs.ndata['feat'].to(device)
#             batch_e = batch_graphs.edata['feat'].to(device)
#             batch_llms = batch_llms.to(device)
#             batch_labels = batch_labels.to(device)

#             if model.name in ["PRGNN", "LINet"]:
#                 batch_scores, score1, score2 = model.forward(batch_graphs, batch_x, batch_e)
#                 loss = model.loss(batch_scores, batch_labels, score1, score2)
#             elif model.name in ['MMbrainG']:
#                 z1, z2, batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_llms)
#                 loss = model.loss(z1, z2, batch_scores, batch_labels)
#             else:
#                 batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_llms)
#                 loss = model.loss(batch_scores, batch_labels)

#                 # --- Compute saliency ---
#                 ig = IntegratedGradients(partial(saliency_forward, e=batch_e, llm=batch_llms, g=batch_graphs))
#                 saliency = ig.attribute(batch_x, n_steps=8, target=0)
#                 if saliency.dim() == 2:
#                     saliency = saliency.view(-1, 116, saliency.size(-1))
#                 roi_saliency = saliency.abs().mean(dim=-1)   # (B, 116)
#                 maps.append(roi_saliency)

#             labels.append(batch_labels.detach().cpu())
#             y_pred.append(batch_scores.detach().cpu())
#             epoch_test_loss += loss.detach().item()
#             epoch_test_acc += accuracy(batch_scores, batch_labels)
#             nb_data += batch_labels.size(0)

#         epoch_test_loss /= (iter + 1)
#         epoch_test_acc /= nb_data

#         y_true = torch.cat(labels, dim=0).numpy()
#         y_pred = torch.cat(y_pred, dim=0).numpy()

#         if len(np.unique(y_true)) == 2:
#             test_precision = precision(y_pred, y_true)
#             test_recall = recall(y_pred, y_true)
#             test_f1 = f1(y_pred, y_true)
#             test_roc_auc = roc_auc(y_pred, y_true)
#         else:
#             test_precision = epoch_test_acc
#             test_recall = epoch_test_acc
#             test_f1 = epoch_test_acc
#             test_roc_auc = epoch_test_acc

#         all_acc = accuracy_all_classes(y_pred, y_true)

#         # --- Compute Top-10 ROIs ---
#         if maps:
#             maps = torch.cat(maps, dim=0)   # (N_samples, 116)
#             roi_score = maps.mean(dim=0)    # (116,)
#             topk = torch.topk(roi_score, k=10)
#             topk_indices = topk.indices.cpu().numpy()
#             topk_scores = topk.values.cpu().numpy()

#             # Save to brain.txt
#             save_path = 'brain.txt'
#             with open(save_path, 'w') as f:
#                 for i, (idx, score) in enumerate(zip(topk_indices, topk_scores)):
#                     f.write(f'{i+1}: ROI {idx}, Score = {score:.6f}\n')

#             print('Top-10 ROI indices:', topk_indices)
#             print('Top-10 ROI scores:', topk_scores)
#             print(f"Results saved to {save_path}")

#     return epoch_test_loss, epoch_test_acc, test_precision, test_recall, test_f1, test_roc_auc, all_acc
import torch
import numpy as np
from functools import partial
from captum.attr import IntegratedGradients

# def evaluate_network_all_metric(model, device, data_loader, epoch=0):
#     """
#     Evaluate network metrics and compute ROI-level saliency scores.
#     Top-10 ROIs will be saved in brain.txt
#     """
#     model.eval()
#     epoch_test_loss = 0
#     epoch_test_acc = 0
#     nb_data = 0

#     labels = []
#     y_pred = []
#     maps = []

#     # -----------------------------
#     # -----------------------------
#     def saliency_forward(x, e, llm, g):
#         """
#         x can be:
#         - (B*116, 116)           -> normal / IG
#         - (B*116, 928)           -> IG expanded
#         """

#         # ----------------------------
#         # ----------------------------
#         if x.dim() == 3:
#             # (B, 116, D) -> (B*116, D)
#             x = x.view(-1, x.size(-1))

#         # ----------------------------
#         # ----------------------------
#         num_nodes = 116
#         total_nodes, feat_dim = x.shape
#         B = total_nodes // num_nodes

#         assert total_nodes % num_nodes == 0, \
#             f"Invalid node number: {total_nodes}"

#         # ----------------------------
#         # ----------------------------
#         if feat_dim != 116:
#             # feat_dim = 116 * n_steps
#             x = x.view(B, num_nodes, -1, 116).mean(dim=2)
#         else:
#             x = x.view(B, num_nodes, 116)

#         # ----------------------------
#         # ----------------------------
#         return model.forward(g, x, e, llm)

#     with torch.no_grad():
#         for iter, (batch_graphs, batch_labels, batch_llms) in enumerate(data_loader):

#             batch_graphs = batch_graphs.to(device)
#             batch_x = batch_graphs.ndata['feat'].to(device)   # (B, 116, 116)
#             batch_e = batch_graphs.edata['feat'].to(device)
#             batch_llms = batch_llms.to(device)
#             batch_labels = batch_labels.to(device)

#             # -----------------------------
#             # Forward & loss
#             # -----------------------------
#             if model.name in ["PRGNN", "LINet"]:
#                 batch_scores, score1, score2 = model.forward(
#                     batch_graphs, batch_x, batch_e
#                 )
#                 loss = model.loss(batch_scores, batch_labels, score1, score2)

#             elif model.name in ['MMbrainG']:
#                 z1, z2, batch_scores = model.forward(
#                     batch_graphs, batch_x, batch_e, batch_llms
#                 )
#                 loss = model.loss(z1, z2, batch_scores, batch_labels)

#             else:
#                 batch_scores = model.forward(
#                     batch_graphs, batch_x, batch_e, batch_llms
#                 )
#                 loss = model.loss(batch_scores, batch_labels)

#                 # -----------------------------
#                 # Integrated Gradients
#                 # -----------------------------
#                 ig = IntegratedGradients(
#                     partial(
#                         saliency_forward,
#                         e=batch_e,
#                         llm=batch_llms,
#                         g=batch_graphs
#                     )
#                 )

#                 saliency = ig.attribute(
#                     batch_x,
#                     n_steps=8,
#                     target=0
#                 )
#                 # saliency shape: (B, 116, 116)

#                 # ROI-level score
#                 roi_saliency = saliency.abs().mean(dim=-1)  # (B, 116)
#                 maps.append(roi_saliency.detach().cpu())

#             # -----------------------------
#             # Metrics
#             # -----------------------------
#             labels.append(batch_labels.detach().cpu())
#             y_pred.append(batch_scores.detach().cpu())

#             epoch_test_loss += loss.detach().item()
#             epoch_test_acc += accuracy(batch_scores, batch_labels)
#             nb_data += batch_labels.size(0)

#         epoch_test_loss /= (iter + 1)
#         epoch_test_acc /= nb_data

#         y_true = torch.cat(labels, dim=0).numpy()
#         y_pred = torch.cat(y_pred, dim=0).numpy()

#         if len(np.unique(y_true)) == 2:
#             test_precision = precision(y_pred, y_true)
#             test_recall = recall(y_pred, y_true)
#             test_f1 = f1(y_pred, y_true)
#             test_roc_auc = roc_auc(y_pred, y_true)
#         else:
#             test_precision = epoch_test_acc
#             test_recall = epoch_test_acc
#             test_f1 = epoch_test_acc
#             test_roc_auc = epoch_test_acc

#         all_acc = accuracy_all_classes(y_pred, y_true)

#         # -----------------------------
#         # Top-10 ROI
#         # -----------------------------
#         if len(maps) > 0:
#             maps = torch.cat(maps, dim=0)    # (N_samples, 116)
#             roi_score = maps.mean(dim=0)     # (116,)

#             topk = torch.topk(roi_score, k=10)
#             topk_indices = topk.indices.numpy()
#             topk_scores = topk.values.numpy()

#             save_path = 'brain.txt'
#             with open(save_path, 'w') as f:
#                 for i, (idx, score) in enumerate(zip(topk_indices, topk_scores)):
#                     f.write(f'{i+1}: ROI {idx}, Score = {score:.6f}\n')

#             print('Top-10 ROI indices:', topk_indices)
#             print('Top-10 ROI scores:', topk_scores)
#             print(f'Results saved to {save_path}')

#     return (
#         epoch_test_loss,
#         epoch_test_acc,
#         test_precision,
#         test_recall,
#         test_f1,
#         test_roc_auc,
#         all_acc
#     )
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

NUM_ROI = 116
SAVE_PATH = "brain.txt"


def evaluate_network_all_metric(model, device, data_loader, epoch=0):
    """
    Evaluate network metrics and compute ROI-level saliency scores
    using gradient-based attribution (node-feature gradients).

    The top-10 ROIs will be saved to:
        brain.txt
    """
    model.eval()

    epoch_test_loss = 0.0
    epoch_test_acc = 0.0
    nb_data = 0

    all_preds = []
    all_labels = []
    maps = []   # (num_samples, 116)

    for it, (batch_graphs, batch_labels, batch_llms) in enumerate(data_loader):

        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)
        batch_llms = batch_llms.to(device)

        batch_x = batch_graphs.ndata["feat"].to(device)
        batch_e = batch_graphs.edata["feat"].to(device)

        # -----------------------------
        # 1. Forward
        # -----------------------------
        batch_x.requires_grad_(True)

        batch_scores = model.forward(
            batch_graphs, batch_x, batch_e, batch_llms
        )

        loss = model.loss(batch_scores, batch_labels)

        # -----------------------------
        # 2. Gradient-based saliency
        # -----------------------------
        model.zero_grad()
        loss.backward(retain_graph=True)

        grad = batch_x.grad.detach()  # (B*N, D) or (B, N, D)

        B = batch_llms.size(0)

        # ---- reshape to (B, 116, D)
        if grad.dim() == 2:
            grad = grad.view(B, NUM_ROI, -1)

        elif grad.dim() != 3:
            raise RuntimeError(f"Unexpected grad shape: {grad.shape}")

        # ---- ROI saliency: average over feature dimension
        roi_saliency = grad.abs().mean(dim=-1)  # (B, 116)

        # ---- safety check
        if roi_saliency.size(1) != NUM_ROI:
            raise RuntimeError(
                f"ROI dim mismatch: expected {NUM_ROI}, got {roi_saliency.size(1)}"
            )

        maps.append(roi_saliency.cpu())

        # -----------------------------
        # 3. Metrics bookkeeping
        # -----------------------------
        all_preds.append(batch_scores.detach().cpu())
        all_labels.append(batch_labels.detach().cpu())

        epoch_test_loss += loss.item()
        epoch_test_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)

    # -----------------------------
    # 4. Aggregate metrics
    # -----------------------------
    epoch_test_loss /= (it + 1)
    epoch_test_acc /= nb_data

    y_true = torch.cat(all_labels, dim=0).numpy()
    y_pred = torch.cat(all_preds, dim=0).numpy()
    y_hat = np.argmax(y_pred, axis=1)

    if len(np.unique(y_true)) == 2:
        test_precision = precision_score(y_true, y_hat, zero_division=0)
        test_recall = recall_score(y_true, y_hat, zero_division=0)
        test_f1 = f1_score(y_true, y_hat, zero_division=0)
        test_roc_auc = roc_auc_score(y_true, y_pred[:, 1])
    else:
        test_precision = epoch_test_acc
        test_recall = epoch_test_acc
        test_f1 = epoch_test_acc
        test_roc_auc = epoch_test_acc

    all_acc = accuracy_all_classes(y_pred, y_true)

    # -----------------------------
    # 5. Top-10 ROI analysis
    # -----------------------------
    maps = torch.cat(maps, dim=0)  # (N_samples, 116)

    if maps.dim() != 2 or maps.size(1) != NUM_ROI:
        raise RuntimeError(f"Invalid maps shape: {maps.shape}")

    roi_score = maps.mean(dim=0)  # (116,)

    topk = torch.topk(roi_score, k=10)
    topk_indices = topk.indices.cpu().numpy()
    topk_scores = topk.values.cpu().numpy()

    with open(SAVE_PATH, "w") as f:
        for i, (idx, score) in enumerate(zip(topk_indices, topk_scores)):
            f.write(f"{i+1}: ROI {idx}, Score = {score:.6f}\n")

    print("Top-10 ROI indices:", topk_indices)
    print("Top-10 ROI scores:", topk_scores)
    print(f"Results saved to {SAVE_PATH}")

    return (
        epoch_test_loss,
        epoch_test_acc,
        test_precision,
        test_recall,
        test_f1,
        test_roc_auc,
        all_acc,
    )
