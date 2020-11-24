import math, random, argparse, time, uuid
import os, os.path as osp
from os.path import join
from helper import makeDirectory, set_gpu
from dataset import DiagDataset

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import KFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch_geometric.datasets import TUDataset
from asap_pool_model import ASAP_Pool

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

torch.backends.cudnn.benchmark = False

from utils import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp


class Trainer(object):

    def __init__(self, params):
        self.p = params

        # set GPU
        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        # build the data
        self.p.use_node_attr = (self.p.dataset == 'FRANKENSTEIN')
        # self.loadData()
        self.load_self_data()

        # build the model
        self.model = None
        self.optimizer = None

    # load data
    def loadData(self):
        path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', self.p.dataset)
        dataset = TUDataset(path, self.p.dataset, use_node_attr=self.p.use_node_attr)
        dataset.data.edge_attr = None
        self.data = dataset

    def load_self_data(self):
        file_dir = join(settings.DATA_DIR, self.p.dataset)
        dataset = DiagDataset(root=file_dir)
        dataset.data.edge_attr = None
        self.data = dataset

    # load model
    def addModel(self):
        if self.p.model == 'ASAP_Pool':
            model = ASAP_Pool(
                dataset=self.data,
                num_layers=self.p.num_layers,
                hidden=self.p.hid_dim,
                ratio=self.p.ratio,
                dropout_att=self.p.dropout_att,
            )

        else:
            raise NotImplementedError
        model.to(self.device).reset_parameters()
        return model

    def addOptimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)

    # train model for an epoch
    def run_epoch(self, loader):
        self.model.train()

        total_loss = 0
        for d_i, data in enumerate(loader):
            self.optimizer.zero_grad()
            data = data.to(self.device)
            ground_truth = data.y.clone()
            out = self.model(data)
            loss = F.nll_loss(out, ground_truth.view(-1))
            loss.backward()
            total_loss += loss.item() * self.num_graphs(data)
            self.optimizer.step()
            if d_i % 20 == 0:
                logger.info("train batch %d", d_i)
        return total_loss / len(loader.dataset)

    # validate or test model
    def predict(self, loader):
        self.model.eval()

        correct = 0
        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                pred = self.model(data).max(1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()

        return correct / len(loader.dataset)

    def evaluate(self, loader, thr=None, return_best_thr=False):
        self.model.eval()

        correct = 0
        total = 0.
        loss, prec, rec, f1 = 0., 0., 0., 0.
        y_true, y_pred, y_score = [], [], []
        for d_i, data in enumerate(loader):
            data = data.to(self.device)
            bs = data.y.size(0)

            with torch.no_grad():
                # pred = self.model(data).max(1)[1]
                out = self.model(data)
                pred = out.max(1)[1]

            loss += F.nll_loss(out, data.y, reduction='sum').item()

            y_true += data.y.data.tolist()
            y_pred += out.max(1)[1].data.tolist()
            y_score += out[:, 1].data.tolist()
            total += bs

            correct += pred.eq(data.y.view(-1)).sum().item()
            if d_i % 50 == 0:
                logger.info("eval batch %d", d_i)

        if thr is not None:
            logger.info("using threshold %.4f", thr)
            y_score = np.array(y_score)
            y_pred = np.zeros_like(y_score)
            y_pred[y_score > thr] = 1

        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
        auc = roc_auc_score(y_true, y_score)
        logger.info("loss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
                    loss / total, auc, prec, rec, f1)

        if return_best_thr:
            precs, recs, thrs = precision_recall_curve(y_true, y_score)
            f1s = 2 * precs * recs / (precs + recs)
            f1s = f1s[:-1]
            thrs = thrs[~np.isnan(f1s)]
            f1s = f1s[~np.isnan(f1s)]
            best_thr = thrs[np.argmax(f1s)]
            logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
            return [prec, rec, f1, auc], loss / len(loader.dataset), best_thr
        else:
            return [prec, rec, f1, auc], loss / len(loader.dataset), None

        # return correct / len(loader.dataset)

    # save model locally
    def save_model(self, save_path):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, save_path)

    # load model from path
    def load_model(self, load_path):
        state = torch.load(load_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])

    # use 10 fold cross-validation
    def k_fold(self):
        kf = KFold(self.p.folds, shuffle=True, random_state=self.p.seed)

        test_indices, train_indices = [], []
        for _, idx in kf.split(torch.zeros(len(self.data)), self.data.data.y):
            test_indices.append(torch.from_numpy(idx))

        val_indices = [test_indices[i - 1] for i in range(self.p.folds)]

        for i in range(self.p.folds):
            train_mask = torch.ones(len(self.data), dtype=torch.uint8)
            train_mask[test_indices[i]] = 0
            train_mask[val_indices[i]] = 0
            train_indices.append(train_mask.nonzero().view(-1))

        return train_indices, test_indices, val_indices

    def num_graphs(self, data):
        if data.batch is not None:
            return data.num_graphs
        else:
            return data.x.size(0)

    # main function for running the experiments
    def run(self):
        val_accs, test_accs = [], []

        makeDirectory('torch_saved/')
        save_path = 'torch_saved/{}'.format(self.p.name)

        if self.p.restore:
            self.load_model(save_path)
            print('Successfully Loaded previous model')

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # iterate over 10 folds
        for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*self.k_fold())):

            # Reinitialise model and optimizer for each fold
            self.model = self.addModel()
            self.optimizer = self.addOptimizer()

            train_dataset = self.data[train_idx]
            test_dataset = self.data[test_idx]
            val_dataset = self.data[val_idx]

            if 'adj' in train_dataset[0]:
                train_loader = DenseLoader(train_dataset, self.p.batch_size, shuffle=True)
                val_loader = DenseLoader(val_dataset, self.p.batch_size, shuffle=False)
                test_loader = DenseLoader(test_dataset, self.p.batch_size, shuffle=False)
            else:
                train_loader = DataLoader(train_dataset, self.p.batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, self.p.batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, self.p.batch_size, shuffle=False)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            best_val_acc, best_test_acc = 0.0, 0.0

            for epoch in range(1, self.p.max_epochs + 1):
                train_loss = self.run_epoch(train_loader)
                val_acc = self.predict(val_loader)

                # lr_decay
                if epoch % self.p.lr_decay_step == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.p.lr_decay_factor * param_group['lr']
                # save model for best val score
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model(save_path)

                print('---[INFO]---{:02d}/{:03d}: Loss: {:.4f}\tVal Acc: {:.4f}'.format(fold + 1, epoch, train_loss,
                                                                                        best_val_acc))

            # load best model for testing
            self.load_model(save_path)
            best_test_acc = self.predict(test_loader)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            val_accs.append(best_val_acc)
            test_accs.append(best_test_acc)

        val_acc_mean = np.round(np.mean(val_accs), 4)
        test_acc_mean = np.round(np.mean(test_accs), 4)

        print('---[INFO]---Val Acc: {:.4f}, Test Accuracy: {:.3f}'.format(val_acc_mean, test_acc_mean))

        return val_acc_mean, test_acc_mean

    def run_new(self):
        val_accs, test_accs = [], []

        makeDirectory('torch_saved/')
        save_path = 'torch_saved/{}'.format(self.p.name)

        if self.p.restore:
            self.load_model(save_path)
            print('Successfully Loaded previous model')

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Reinitialise model and optimizer for each fold
        self.model = self.addModel()
        self.optimizer = self.addOptimizer()

        dataset = self.data

        num_training = int(len(dataset) * 0.5)
        num_val = int(len(dataset) * 0.75) - num_training
        num_test = len(dataset) - (num_training + num_val)
        # training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
        train_dataset = dataset[:num_training]
        val_dataset = dataset[num_training:(num_training + num_val)]
        test_dataset = dataset[(num_training + num_val):]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, self.p.batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, self.p.batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, self.p.batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, self.p.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, self.p.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, self.p.batch_size, shuffle=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        best_val_acc, best_test_acc = 0.0, 0.0
        best_thr = None

        val_metrics, val_loss, thr = self.evaluate(val_loader, return_best_thr=True)
        test_metrics, test_loss, _ = self.evaluate(test_loader, thr=0.5)

        for epoch in range(1, self.p.max_epochs + 1):
            train_loss = self.run_epoch(train_loader)
            val_metrics, val_loss, thr = self.evaluate(val_loader, return_best_thr=True)
            test_metrics, test_loss, _ = self.evaluate(test_loader, thr=thr)
            val_auc = val_metrics[-1]

            # lr_decay
            if epoch % self.p.lr_decay_step == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.p.lr_decay_factor * param_group['lr']
            # save model for best val score
            if val_auc > best_val_acc:
                best_val_acc = val_auc
                best_thr = thr
                self.save_model(save_path)

            print('---[INFO]---{:03d}: Loss: {:.4f}\tVal Acc: {:.4f}'.format(epoch, train_loss, best_val_acc))
            print('---[INFO]---{:03d}: Test metrics'.format(epoch), test_metrics)

        # load best model for testing
        self.load_model(save_path)
        test_metrics, test_loss, _ = self.evaluate(test_loader, thr=thr)
        print('---[INFO]---{:03d}: Test metrics'.format(epoch), test_metrics)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Neural Network Trainer Template')

    parser.add_argument('-model', dest='model', default='ASAP_Pool', help='Model to use')
    parser.add_argument('-data', dest='dataset', default='twitter', type=str, help='Dataset to use')
    parser.add_argument('-epoch', dest='max_epochs', default=100, type=int, help='Max epochs')
    parser.add_argument('-l2', dest='l2', default=5e-4, type=float, help='L2 regularization')
    parser.add_argument('-num_layers', dest='num_layers', default=3, type=int, help='Number of GCN Layers')
    parser.add_argument('-lr_decay_step', dest='lr_decay_step', default=50, type=int, help='lr decay step')
    parser.add_argument('-lr_decay_factor', dest='lr_decay_factor', default=0.5, type=float, help='lr decay factor')

    parser.add_argument('-batch', dest='batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('-hid_dim', dest='hid_dim', default=64, type=int, help='hidden dims')
    parser.add_argument('-dropout_att', dest='dropout_att', default=0.1, type=float, help='dropout on attention scores')
    parser.add_argument('-lr', dest='lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('-ratio', dest='ratio', default=0.5, type=float, help='ratio')

    parser.add_argument('-folds', dest='folds', default=10, type=int, help='Cross validation folds')

    parser.add_argument('-name', dest='name', default='test_' + str(uuid.uuid4())[:8], help='Name of the run')
    parser.add_argument('-gpu', dest='gpu', default='1', help='GPU to use')
    parser.add_argument('-restore', dest='restore', action='store_true', help='Model restoring')

    args = parser.parse_args()
    if not args.restore:
        args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

    print('Starting runs...')
    print(args)

    # get 20 run average
    seeds = [8971, 85688, 9467, 32830, 28689, 94845, 69840, 50883, 74177, 79585, 1055, 75631, 6825, 93188, 95426, 54514,
             31467, 70597, 71149, 81994]
    seeds = [42]
    counter = 0
    args.log_db = args.name
    print("log_db:", args.log_db)
    avg_val = []
    avg_test = []
    for seed in seeds:
        # set seed
        args.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        set_gpu(args.gpu)

        args.name = '{}_run_{}'.format(args.log_db, counter)

        # start training the model
        model = Trainer(args)
        # val_acc, test_acc = model.run()
        model.run_new()
        # print('For seed {}\t Val Accuracy: {:.3f} \t Test Accuracy: {:.3f}\n'.format(seed, val_acc, test_acc))
        # avg_val.append(val_acc)
        # avg_test.append(test_acc)
        counter += 1

    # print('Val Accuracy: {:.3f} ± {:.3f} Test Accuracy: {:.3f} ± {:.3f}'.format(np.mean(avg_val), np.std(avg_val),
    #                                                                             np.mean(avg_test), np.std(avg_test)))
