import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph
import numpy as np
import scipy.sparse as sp

EPS = 1e-15

class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dropout=0.):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        input = F.dropout(x, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class VGAE(nn.Module):
    """
    The self-supervised module of DeepDSI
    """
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(VGAE, self).__init__()
        self.gc1 = GCNLayer(input_feat_dim, hidden_dim1, dropout)   # F.relu
        self.gc2 = GCNLayer(hidden_dim1, hidden_dim2, dropout)    # lambda x: x
        self.gc3 = GCNLayer(hidden_dim1, hidden_dim2, dropout)
        self.act1 = nn.ReLU()

    def encode(self, x, adj):
        hidden1 = self.act1(self.gc1(x, adj))
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        adj_hat = torch.mm(z, z.t())
        return adj_hat

    def forward(self, x, adj, sigmoid: bool = True):
        mu, logstd = self.encode(x, adj)
        z = self.reparameterize(mu, logstd)
        return (torch.sigmoid(self.decode(z)), z, mu, logstd) if sigmoid else (self.decode(z), z, mu, logstd)


class DSIPredictor(nn.Module):
    """
    The semi-supervised module of DeepDSI
    """
    def __init__(self, in_features, out_features):
        super(DSIPredictor, self).__init__()

        self.gc1 = GCNLayer(343, 343, 0.1)
        self.gc2 = GCNLayer(343, 343, 0.1)

        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, out_features)
        self.dropout = nn.Dropout(p=0.4)
        self.act1 = nn.ReLU()
        self.act2 = nn.Mish()

    def forward(self, x, adj, pro1_index, pro2_index, sigmoid: bool = True):
        x1 = self.act1(self.gc1(x, adj))
        x2 = self.gc2(x1, adj)

        pro1 = x2[pro1_index]
        pro2 = x2[pro2_index]

        dsi = torch.cat([pro1, pro2], dim = 1)

        h1 = self.dropout(self.bn1(self.act2(self.fc1(dsi))))
        h2 = self.dropout(self.bn2(self.act2(self.fc2(h1))))
        h3 = self.dropout(self.bn3(self.act2(self.fc3(h2))))
        h4 = self.fc4(h3)

        return torch.sigmoid(h4) if sigmoid else h4


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Linear(20398, 343)
        self.decoder = nn.Linear(343, 20398)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Classifier(nn.Module):
    # __constants__ = ['in_features', 'out_features']
    def __init__(self, in_features, out_features):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, out_features)
        self.dropout = nn.Dropout(p=0.4)
        self.act = nn.Mish()

    def forward(self, x, pro1_index, pro2_index):
        pro1 = x[pro1_index]
        pro2 = x[pro2_index]

        x = torch.cat([pro1, pro2], dim = 1)

        x = self.dropout(self.bn1(self.act(self.fc1(x))))
        x = self.dropout(self.bn2(self.act(self.fc2(x))))
        x = self.dropout(self.bn3(self.act(self.fc3(x))))
        x = torch.sigmoid(self.fc4(x))

        return x


class PairExplainer(torch.nn.Module):
    """The explainable module of DeepDSI

    Args:
        model (torch.nn.Module): The module to importance.
        lr (float, optional): The learning rate to apply.
        num_hops (int, optional): The number of hops the 'model' is aggregating information from.
        feat_mask_obj (str, optional): Denotes the object of feature mask that will be learned.
        log (bool, optional): Choose whether to log learning progress.
        **kwargs (optional): Additional hyper-parameters to override default settings of the 'coeffs'.
    """

    coeffs = {
        'feat_size': 1.0,
        'feat_reduction': 'mean',
        'feat_ent': 0.5,   # 0.1
    }

    def __init__(self, model, lr: float = 0.01,
                 num_hops: int = 2,
                 feat_mask_obj: str = 'dsi',
                 log: bool = True, **kwargs):
        super().__init__()
        assert feat_mask_obj in ['dub', 'sub', 'dsi']
        self.model = model
        self.lr = lr
        self.num_hops = num_hops
        self.feat_mask_obj = feat_mask_obj
        self.log = log
        self.coeffs.update(kwargs)
        self.device = next(model.parameters()).device

    def __set_masks__(self, num_feat):
        std = 0.1
        if self.feat_mask_obj == 'dsi':
            self.feat_mask = torch.nn.Parameter(torch.randn(2, num_feat) * std)
        else:
            self.feat_mask = torch.nn.Parameter(torch.randn(1, num_feat) * std)

    def __clear_masks__(self):
        self.feat_masks = None

    def __subgraph__(self, node_idx, x, edge_index):
        # covert the type of edge_index
        edge_index = edge_index.cpu().detach().to_dense()
        tmp_coo = sp.coo_matrix(edge_index)
        edge_index = np.vstack((tmp_coo.row, tmp_coo.col))
        edge_index = torch.LongTensor(edge_index)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=x.size(0), flow='source_to_target')

        return subset

    def __loss__(self, log_logits, pred_label):
        loss1 = torch.cdist(log_logits, pred_label)
        m = self.feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['feat_reduction'])
        loss2 = self.coeffs['feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss3 =  self.coeffs['feat_ent'] * ent.mean()

        return loss1 + loss2 + loss3

    def explain(self, x, adj, pro1_index, pro2_index, epochs: int = 100):
        """Learn and return a feature mask that explains the importance of each dimension of the feature

        Args:
            x (Tensor): The node feature matrix.
            adj (Tensor): The adjacency matrix.
            pro1_index (int): The protein1 to importance.
            pro2_index (int): The protein2 to importance.
            epochs (int, optional): The number of epochs to train.

        rtype: (Tensor)
        """

        self.model.eval()
        self.__clear_masks__()

        # 1. Get the subgraphs.
        subset1 = self.__subgraph__(pro1_index, x, adj)
        subset2 = self.__subgraph__(pro2_index, x, adj)

        # 2. Get the initial prediction.
        with torch.no_grad():
            pred_label = self.model(x, adj, [pro1_index], [pro2_index])

        # 3. Initialize the weight
        self.__set_masks__(x.size(1))
        self.to(self.device)


        parameters = [self.feat_mask]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        if self.log:
            pbar = tqdm(total=epochs)
            pbar.set_description(f'importance this pair of DSI')

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            # 4. Use the weight
            h = x.clone()
            if self.feat_mask_obj == 'dub':
                h[subset1] = h[subset1].clone() * self.feat_mask.sigmoid()
            if self.feat_mask_obj == 'sub':
                h[subset2] = h[subset2].clone() * self.feat_mask.sigmoid()
            if self.feat_mask_obj == 'dsi':
                h[subset1] = h[subset1].clone() * self.feat_mask[0].sigmoid()
                h[subset2] = h[subset2].clone() * self.feat_mask[1].sigmoid()

            log_logits = self.model(h, adj, [pro1_index], [pro2_index])

            loss = self.__loss__(log_logits, pred_label)
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        feat_mask = self.feat_mask.detach().sigmoid().cpu()
        self.__clear_masks__()

        return feat_mask
