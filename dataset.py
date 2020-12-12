from os.path import join, isfile
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


def load_w2v_feature(file, max_idx=0):
    with open(file, "rb") as f:
        nu = 0
        for line in f:
            content = line.strip().split()
            nu += 1
            if nu == 1:
                n, d = int(content[0]), int(content[1])
                feature = [[0.] * d for i in range(max(n, max_idx + 1))]
                continue
            index = int(content[0])
            while len(feature) <= index:
                feature.append([0.] * d)
            for i, x in enumerate(content[1:]):
                feature[index][i] = float(x)
    for item in feature:
        assert len(item) == d
    return np.array(feature, dtype=np.float32)


class DiagDataset(InMemoryDataset):
    def __init__(self, root, label_type="like", transform=None, pre_transform=None, seed=42):
        self.file_dir = root
        self.seed = seed
        self.label_type = label_type
        super(DiagDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print("shape", self.data.x.shape)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [join(self.file_dir, "diag_{}.dat".format(self.label_type))]

    def download(self):
        pass

    def get_samples_num(self, role):
        if settings.TEST_SIZE < np.iinfo(np.int64).max:
            return int(settings.TEST_SIZE/3)
        file_dir = self.file_dir
        labels = np.load(join(file_dir, "{}_{}_labels.npy".format(role, self.label_type)))
        return len(labels)

    def process(self):
        data_list = []
        file_dir = self.file_dir

        dataset = self.file_dir.split("/")[-1]
        if dataset != "wechat":
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

            embedding_path = join(file_dir, "prone.emb2")
            max_vertex_idx = np.max(vertices)
            embedding = load_w2v_feature(embedding_path, max_vertex_idx)
            # self.embedding = torch.FloatTensor(embedding)
            logger.info("%d-dim embedding loaded!", embedding[0].shape[0])

            n_graphs = len(graphs)

            for i, adj in enumerate(graphs):
                if i % 10000 == 0:
                    logger.info("process graph %d", i)
                cur_vids = vertices[i]
                cur_node_features = vertex_features[cur_vids]
                cur_emb = embedding[cur_vids]
                adj_coo = coo_matrix(np.asmatrix(adj))
                edge_index = torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long)
                inf_features = influence_features[i]
                x = np.concatenate((cur_node_features, inf_features, cur_emb),
                                   axis=1)  # todo global network embedding features
                x = torch.FloatTensor(x)
                y = torch.LongTensor([labels[i]])
                data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(data)

                if i > settings.TEST_SIZE:
                    break

            data_list = sklearn.utils.shuffle(data_list, random_state=self.seed)

        else:
            embedding = np.empty(shape=(0, 64))
            if isfile(join(file_dir, "node_embedding_spectral.npy")):
                embedding = np.load(join(file_dir, "node_embedding_spectral.npy"))
                logger.info("%d-dim embedding loaded!", embedding[0].shape[0])
            else:
                # embedding = np.load(os.path.join(settings.DATA_DIR, "node_embedding_spectral.npy"))
                for emb_i in range(5):
                    # with np.load(join(settings.DATA_DIR, "node_embedding_spectral_{}.npz".format(emb_i))) as data:
                    data = np.load(join(file_dir, "node_embedding_spectral_{}.npz".format(emb_i)))
                    embedding = np.concatenate((embedding, data["emb"]))
                    logger.info("load emb batch %d", emb_i)
                    del data
            tmp = np.zeros((64,))
            embedding = np.row_stack((embedding, tmp))
            # self.embedding = torch.FloatTensor(embedding)

            # del embedding

            vertex_features = np.load(join(file_dir, "user_features.npy"))
            vertex_features = preprocessing.scale(vertex_features)
            vertex_features = np.concatenate((vertex_features,
                                              np.zeros(shape=(1, vertex_features.shape[1]))), axis=0)
            logger.info("global vertex features loaded!")

            graphs_train = np.load(join(file_dir, "train_adjacency_matrix.npy"))
            logger.info("train graphs loaded")
            graphs_valid = np.load(join(file_dir, "valid_adjacency_matrix.npy"))
            logger.info("valid graphs loaded")
            graphs_test = np.load(join(file_dir, "test_adjacency_matrix.npy"))
            logger.info("test graphs loaded")

            graphs = np.vstack((graphs_train, graphs_valid, graphs_test))
            logger.info("all graphs got")
            print("graphs shape", graphs.shape)

            del graphs_train, graphs_valid, graphs_test

            # roles = ["train", "valid", "test"]
            # for role in roles:
            inf_features_train = np.load(join(file_dir, "train_influence_features.npy")).astype(np.float32)
            logger.info("influence features train loaded!")
            inf_features_valid = np.load(join(file_dir, "valid_influence_features.npy")).astype(np.float32)
            logger.info("influence features valid loaded!")
            inf_features_test = np.load(join(file_dir, "test_influence_features.npy")).astype(np.float32)
            logger.info("influence features test loaded!")

            inf_features = np.vstack((inf_features_train, inf_features_valid, inf_features_test))
            logger.info("inf features got")

            del inf_features_train, inf_features_valid, inf_features_test

            labels_train = np.load(join(file_dir, "train_{}_labels.npy".format(self.label_type)))
            logger.info("labels train loaded!")
            labels_valid = np.load(join(file_dir, "valid_{}_labels.npy".format(self.label_type)))
            logger.info("labels valid loaded!")
            labels_test = np.load(join(file_dir, "test_{}_labels.npy".format(self.label_type)))
            logger.info("labels test loaded!")

            labels = np.concatenate((labels_train, labels_valid, labels_test))
            logger.info("labels loaded")

            vertices_train = np.load(join(file_dir, "train_vertex_ids.npy"))
            logger.info("vertex ids train loaded!")
            vertices_valid = np.load(join(file_dir, "valid_vertex_ids.npy"))
            logger.info("vertex ids valid loaded!")
            vertices_test = np.load(join(file_dir, "test_vertex_ids.npy"))
            logger.info("vertex ids test loaded!")

            vertices = np.vstack((vertices_train, vertices_valid, vertices_test))
            logger.info("vertex ids got")
            del vertices_train, vertices_valid, vertices_test

            for i, adj in enumerate(graphs):
                if i % 10000 == 0:
                    logger.info("process graph %d", i)
                cur_vids = vertices[i]
                cur_node_features = vertex_features[cur_vids]
                cur_emb = embedding[cur_vids]
                adj_coo = coo_matrix(np.asmatrix(adj))
                edge_index = torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long)
                cur_inf_features = inf_features[i]
                x = np.concatenate((cur_node_features, cur_inf_features, cur_emb),
                                   axis=1)  # todo global network embedding features
                x = torch.FloatTensor(x)
                y = torch.LongTensor([labels[i]])
                data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(data)

                if i > settings.TEST_SIZE:
                    break

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
