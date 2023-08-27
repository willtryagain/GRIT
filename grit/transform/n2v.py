# ------------------------ : new rwpse ----------------
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torch_sparse
from torch_geometric.data import Data, HeteroData
from torch_geometric.graphgym.config import cfg
from torch_geometric.nn import Node2Vec
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    to_scipy_sparse_matrix,
)
from icecream import ic
from torch_scatter import scatter, scatter_add, scatter_max
from torch_sparse import SparseTensor


def add_node_attr(data: Data, value: Any, attr_name: Optional[str] = None) -> Data:
    if attr_name is None:
        if "x" in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data


@torch.no_grad()
def add_full_n2v(
    data,
    walk_length=8,
    attr_name_abs="n2v",  # name: 'rrwp'
    attr_name_rel="n2v",  # name: ('rrwp_idx', 'rrwp_val')
    add_identity=True,
    spd=False,
    **kwargs,
):
    device = data.edge_index.device
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight
    model = Node2Vec(
        edge_index,
        embedding_dim=walk_length,
        walk_length=walk_length,
    ).to(device)
    num_workers = 4
    loader = model.loader(batch_size=128, shuffle=True,
                          num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=150)
        return acc

    for epoch in range(1, 101):
        loss = train()
        acc = test()
    
    z = model()
    # check if z has shape (num_nodes, walk_length)
    ic(z.size())
    data = add_node_attr(data, z, attr_name=attr_name_abs)

    # construct pairwise hadamard product of embeddings
    prod = torch.einsum("ij, kj -> ikj", z, z)
    # find indices of non-zero elements and their values
    idx = prod.nonzero(as_tuple=True)
    val = prod[idx]
    ic(idx.shape, val.shape)
    exit()
    # pair wise hadamard

    return data
