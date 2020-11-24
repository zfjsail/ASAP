from os.path import join
import numpy as np
import sklearn
from sklearn import preprocessing
from scipy.sparse import coo_matrix
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from utils import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


class DiagDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, seed=42):
        self.file_dir = root
        self.seed = seed
        super(DiagDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print("shape", self.data.x.shape)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [join(self.file_dir, "diag.dat")]

    def download(self):
        pass

    def process(self):
        data_list = []
        file_dir = self.file_dir

        graphs = np.load(join(file_dir, "adjacency_matrix.npy")).astype(np.float32)

        # wheather a user has been influenced
        # wheather he/she is the ego user
        influence_features = np.load(
                join(file_dir, "influence_feature.npy")).astype(np.float32)
        logger.info("influence features loaded!")

        labels = np.load(join(file_dir, "label.npy"))
        logger.info("labels loaded!")

        vertices = np.load(join(file_dir, "vertex_id.npy"))
        logger.info("vertex ids loaded!")

        vertex_features = np.load(join(file_dir, "vertex_feature.npy"))
        vertex_features = preprocessing.scale(vertex_features)
        # vertex_features = torch.FloatTensor(vertex_features)
        logger.info("global vertex features loaded!")

        n_graphs = len(graphs)

        for i, adj in enumerate(graphs):
            if i % 10000 == 0:
                logger.info("process graph %d", i)
            cur_vids = vertices[i]
            cur_node_features = vertex_features[cur_vids]
            adj_coo = coo_matrix(np.asmatrix(adj))
            edge_index = torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long)
            inf_features = influence_features[i]
            x = np.concatenate((cur_node_features, inf_features), axis=1)  # todo global network embedding features
            x = torch.FloatTensor(x)
            y = torch.LongTensor([labels[i]])
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data_list = sklearn.utils.shuffle(data_list, random_state=self.seed)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

