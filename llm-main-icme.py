import numpy as np
import os
import time
import random
import glob
import argparse, json
import dgl
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from nets.load_net import gnn_model
from data.data import LoadData_llm
from trainer import evaluate_network_all_metric
from trainer import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network
from utils import _logger


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def view_model_param(MODEL_NAME, model):
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    return total_param


def train_val_pipeline(MODEL_NAME, DATASET_NAME, dataset, params, net_params, dirs):
    avg_test_acc, avg_test_precision, avg_test_recall, avg_test_f1, avg_test_roc_auc = [], [], [], [], []
    avg_train_acc = []
    avg_convergence_epochs = []
    avg_all_test_acc = torch.zeros(net_params['n_classes'])

    t0 = time.time()
    per_epoch_time = []

    if MODEL_NAME in ['GCN', 'GAT', 'PRGNN', 'GXN', 'BrainGNN']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\n""".format(DATASET_NAME, MODEL_NAME, params, net_params))

    try:
        for split_number in range(1):
            log_file_name = os.path.join(root_log_dir,
                                         f"logs_RUN_{str(split_number+1)}.log")
            logger = _logger(log_file_name)

            # setting seeds
            os.environ['PYTHONHASHSEED'] = str(params['seed'])
            random.seed(params['seed'])
            np.random.seed(params['seed'])
            torch.manual_seed(params['seed'])
            torch.cuda.manual_seed_all(params['seed'])
            dgl.seed(params['seed'])
            dgl.random.seed(params['seed'])

            if device.type == 'cuda':
                torch.cuda.manual_seed(params['seed'])
                torch.cuda.manual_seed_all(params['seed'])
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            logger.error("=" * 45)
            logger.error(f"RUN NUMBER:         {split_number+1}")
            logger.error("=" * 45)
            trainset, valset, testset = dataset.train[split_number], dataset.val[split_number], dataset.test[split_number]
            logger.debug(f"Training Graphs:    {len(trainset)}")
            logger.debug(f"Validation Graphs:  {len(valset)}")
            logger.debug(f"Test Graphs:        {len(testset)}")
            logger.debug(f"Number of Classes:  {net_params['n_classes']}")
            logger.debug("=" * 45)


            if MODEL_NAME in ['BrainPromptG', 'BrainPromptC', 'MMbrainG', 'BrainPromptMSHT']:
                prompt_name = DATASET_NAME.split('_')[0] + '_label.pt'
                label_path = 'data/prompts/label_prompts/' + prompt_name
                if os.path.exists(label_path):
                    label_embs = torch.stack(torch.load(label_path)).squeeze()
                    net_params['label_embs'] = label_embs
                else:
                    logger.warning(f"Label prompt file not found at {label_path}")

            model = gnn_model(MODEL_NAME, net_params, trainset)
            model = model.to(device)

            net_params['total_param'] = view_model_param(MODEL_NAME, model)
            logger.debug('MODEL/Total parameters: %s, %d', MODEL_NAME, net_params['total_param'])

            optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=params['lr_reduce_factor'],
                                                             patience=params['lr_schedule_patience'])

            epoch_train_losses, epoch_val_losses = [], []
            epoch_train_accs, epoch_val_accs = [], []

            drop_last = True if MODEL_NAME in ['DiffPool'] else False

            train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
            val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
            test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)

            best_test_acc = 0.0
            with tqdm(range(params['epochs'])) as t:
                for epoch in t:
                    t.set_description('Epoch %d' % epoch)
                    start = time.time()

                    if MODEL_NAME in ['RingGNN', '3WLGNN']:
                        epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, params['batch_size'])
                    else:
                        epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)

                    epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch)
                    _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch)

                    if epoch_test_acc > best_test_acc:
                        best_test_acc = epoch_test_acc
                        # saliency_path = os.path.join(
                        #     root_log_dir, f'roi_saliency_split{split_number}.pt'
                        # )

                        # _, _, test_precision, test_recall, test_f1, test_roc_auc, all_test_acc=evaluate_network_all_metric(
                        #     model, device, test_loader, epoch, path=saliency_path
                        # )

                        # saliency_path = os.path.join(
                        #     root_log_dir, f'roi_saliency_split{split_number}.pt'
                        # )

                        # _, _, test_precision, test_recall, test_f1, test_roc_auc, all_test_acc = \
                        #     evaluate_network_all_metric(
                        #         model, device, test_loader, epoch, path=saliency_path
                        #     )
                        _, _, test_precision, test_recall, test_f1, test_roc_auc, all_test_acc = evaluate_network_all_metric(
                            model, device, test_loader, epoch)

                        best_test_metrics = {
                            'acc': best_test_acc,
                            'precision': test_precision,
                            'recall': test_recall,
                            'f1': test_f1,
                            'roc_auc': test_roc_auc,
                            'all_acc': all_test_acc
                        }

                    epoch_train_losses.append(epoch_train_loss)
                    epoch_val_losses.append(epoch_val_loss)
                    epoch_train_accs.append(epoch_train_acc)
                    epoch_val_accs.append(epoch_val_acc)

                    t.set_postfix(lr=optimizer.param_groups[0]['lr'],
                                  train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                  train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                                  test_acc=epoch_test_acc, best_acc=best_test_acc)

                    per_epoch_time.append(time.time()-start)
                    scheduler.step(epoch_val_loss)

            _, train_acc = evaluate_network(model, device, train_loader, epoch)
            avg_train_acc.append(train_acc)
            avg_convergence_epochs.append(epoch)

            avg_test_acc.append(best_test_metrics['acc'])
            avg_test_precision.append(best_test_metrics['precision'])
            avg_test_recall.append(best_test_metrics['recall'])
            avg_test_f1.append(best_test_metrics['f1'])
            avg_test_roc_auc.append(best_test_metrics['roc_auc'])

            logger.warning("Test Accuracy [BEST]: {:.4f}".format(best_test_metrics['acc']))
            logger.warning("Train Precision [BEST]: {:.4f}".format(best_test_metrics['precision']))
            logger.warning("Train Recall [BEST]: {:.4f}".format(best_test_metrics['recall']))
            logger.warning("Train F1 [BEST]: {:.4f}".format(best_test_metrics['f1']))
            logger.warning("Train ROC-AUC [BEST]: {:.4f}".format(best_test_metrics['roc_auc']))
            logger.warning("Convergence Time (Epochs): {:.4f}\n".format(epoch))

    except KeyboardInterrupt:
        logger.error("=" * 45)
        logger.error('Exiting from training early because of KeyboardInterrupt')

    logger.debug("TOTAL TIME TAKEN: {:.4f}hrs".format((time.time() - t0) / 3600))
    logger.debug("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    logger.debug("AVG CONVERGENCE Time (Epochs): {:.4f}\n".format(np.mean(np.array(avg_convergence_epochs))))

    logger.debug("""FINAL RESULTS------TRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}""".format(
        np.mean(np.array(avg_train_acc)) * 100, np.std(avg_train_acc) * 100))
    logger.debug("All splits Train Accuracies: %s", avg_train_acc)
    logger.debug("""FINAL RESULTS------TEST ACCURACY averaged: {:.4f} with s.d. {:.4f}""".format(
        np.mean(np.array(avg_test_acc)) * 100, np.std(avg_test_acc) * 100))
    logger.debug("All splits Test Accuracies: %s\n", avg_test_acc)

    logger.debug("""TEST precision averaged: {:.4f} with s.d. {:.4f}""".format(
        np.mean(np.array(avg_test_precision)) * 100, np.std(avg_test_precision) * 100))
    logger.debug("""TEST recall averaged: {:.4f} with s.d. {:.4f}""".format(
        np.mean(np.array(avg_test_recall)) * 100, np.std(avg_test_recall) * 100))
    logger.debug("""TEST f1 averaged: {:.4f} with s.d. {:.4f}""".format(
        np.mean(np.array(avg_test_f1)) * 100, np.std(avg_test_f1) * 100))
    logger.debug("""TEST roc_auc averaged: {:.4f} with s.d. {:.4f}""".format(
        np.mean(np.array(avg_test_roc_auc)) * 100, np.std(avg_test_roc_auc) * 100))

    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY: {:.4f} with s.d. {:.4f}\nTEST precision: {:.4f} with s.d. {:.4f}\nTEST recall: {:.4f} with s.d. {:.4f}\n
    TEST f1: {:.4f} with s.d. {:.4f}\nTEST roc_auc: {:.4f} with s.d. {:.4f}\n\n\nTRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}\n\n
    Average Convergence Time (Epochs): {:.4f} with s.d. {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\nAll Splits Test Accuracies: {}""" \
                .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                        np.mean(np.array(avg_test_acc)) * 100, np.std(avg_test_acc) * 100,
                        np.mean(np.array(avg_test_precision)) * 100, np.std(avg_test_precision) * 100,
                        np.mean(np.array(avg_test_recall)) * 100, np.std(avg_test_recall) * 100,
                        np.mean(np.array(avg_test_f1)) * 100, np.std(avg_test_f1) * 100,
                        np.mean(np.array(avg_test_roc_auc)) * 100, np.std(avg_test_roc_auc) * 100,
                        np.mean(np.array(avg_train_acc))*100, np.std(avg_train_acc)*100,
                        np.mean(avg_convergence_epochs), np.std(avg_convergence_epochs),
                        (time.time()-t0)/3600, np.mean(per_epoch_time), avg_test_acc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--threshold', type=float, help="Please give a threshold to drop edge", default=0.3)
    parser.add_argument('--edge_ratio', type=float, help="Please give a ratio to drop edge", default=0)
    parser.add_argument('--node_feat_transform', help="Please give a value for node feature transform", default=None)
    parser.add_argument('--pos_enc', help="Please give a value for positional encoding")
    parser.add_argument('--contrast', default=False, action='store_true')
    parser.add_argument('--pooling', type=float, default=0.5)
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--learnable_q', default=False, action='store_true')
    parser.add_argument('--dis_loss', default=False, action='store_true')
    parser.add_argument('--spatial', type=str, default='')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None and config['gpu']['use']:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
        device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    else:
        config['gpu']['id'] = 0
        device = torch.device('cpu')
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData_llm(DATASET_NAME, args.threshold, args.edge_ratio, args.node_feat_transform)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    if args.threshold is not None:
        params['threshold'] = float(args.threshold)
    if args.edge_ratio is not None:
        params['edge_ratio'] = float(args.edge_ratio)
    if args.node_feat_transform is not None:
        params['node_feat_transform'] = args.node_feat_transform

    # network parameters
    net_params = config['net_params']
    if 'node_num' in dir(dataset):
        net_params['node_num'] = int(dataset.node_num)
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred=='True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
    if args.contrast is not None:
        net_params['contrast'] = args.contrast
    if args.pooling is not None:
        net_params['pooling'] = float(args.pooling)
    if args.lambda1 is not None:
        net_params['lambda1'] = float(args.lambda1)
    if args.tau is not None:
        net_params['tau'] = float(args.tau)
    if args.alpha is not None:
        net_params['alpha'] = float(args.alpha)
    if args.learnable_q is not None:
        net_params['learnable_q'] = args.learnable_q
    # if args.pos_enc is not None:
    net_params['pos_enc'] = args.pos_enc
    if args.dis_loss is not None:
        net_params['dis_loss'] = args.dis_loss
    if args.spatial is not None:
        params['spatial'] = args.spatial

    # TUs
    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'].shape[1]
    net_params['edge_dim'] = dataset.all.graph_lists[0].edata['feat'][0].shape[0] \
        if 'feat' in dataset.all.graph_lists[0].edata else None
    num_classes = len(np.unique(dataset.all.graph_labels))
    net_params['n_classes'] = num_classes

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Y-%m-%d_%H-%M-%S')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Y-%m-%d_%H-%M-%S')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Y-%m-%d_%H-%M-%S')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Y-%m-%d_%H-%M-%S')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    train_val_pipeline(MODEL_NAME, DATASET_NAME, dataset, params, net_params, dirs)


main()
