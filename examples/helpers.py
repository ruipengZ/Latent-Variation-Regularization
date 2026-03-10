import math
from torch_geometric.utils import degree
from torch_geometric.utils import to_undirected, coalesce
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from torch_cluster import knn_graph
import torch
import torch.nn.functional as F


@torch.no_grad()
def knn_epsilon_graph(
    x: torch.Tensor,
    k: int = 16,
    *,
    metric: str = "euclidean",   # "euclidean" | "cosine"
    q: float = 1,
    undirected: bool = True,
):
    N = x.size(0)
    if metric not in ("euclidean", "cosine"):
        raise ValueError("metric must be 'euclidean' or 'cosine'")

    x_norm = F.normalize(x, dim=-1) if metric == "cosine" else x

    def pair_dist(a, b):
        if metric == "euclidean":
            return (x[a] - x[b]).pow(2).sum(-1).sqrt()
        else:
            return 1.0 - (x_norm[a] * x_norm[b]).sum(-1)

    # Candidate KNN edges
    ei = knn_graph(x_norm, k=k, loop=False, flow="target_to_source")
    s, d = ei
    dist = pair_dist(s, d)

    # Local eos = q-quantile of distances from s to its k neighbors
    order = torch.argsort(s)
    d_mat = dist[order].view(N, k)
    r = max(1, math.ceil(q * k))
    eps = torch.kthvalue(d_mat, k=r, dim=1).values

    # Prune edges
    keep = dist <= eps[s]
    ei = ei[:, keep]

    if undirected:
        ei = to_undirected(ei, num_nodes=N)
    ei, _ = coalesce(ei, None, num_nodes=N)
    return ei



def build_delta_dict(edge_index: torch.Tensor,
                     num_nodes: int,
                     device=None,
                     min_neighbors: int | None = None,
                     max_neighbors: int | None = None):
    # --- 0) Coalesce so ordering matches LineGraph's internal mapping ---
    ei, _ = coalesce(edge_index, None, num_nodes=num_nodes)
    row, col = ei

    # --- 1) Unique undirected edges in the SAME order LineGraph uses ---
    mask = row < col
    i_idx = row[mask].to(device)  # [E_unique]
    j_idx = col[mask].to(device)  # [E_unique]
    E_unique = i_idx.numel()

    # --- 2) Build the PyG line graph (undirected mode) ---
    data = Data(num_nodes=num_nodes, edge_index=edge_index)
    L = LineGraph(force_directed=False)(data)  # edges are directed both ways
    sL, dL = L.edge_index
    assert L.num_nodes == E_unique, f"LineGraph nodes={L.num_nodes} vs E_unique={E_unique}"

    # --- 3) Adjacency list & dataset (anchor -> neighbor list) ---
    adj_list = [[] for _ in range(E_unique)]
    for u, v in zip(sL.tolist(), dL.tolist()):
        if u != v:
            adj_list[u].append(v)

    delta_dataset = []
    for anchor, nbrs in enumerate(adj_list):
        if min_neighbors is not None and len(nbrs) < min_neighbors:
            continue
        if max_neighbors is not None and len(nbrs) > max_neighbors:
            import random
            nbrs = random.sample(nbrs, max_neighbors)
        delta_dataset.append((anchor, nbrs))

    # print(f"Precomputed {len(delta_dataset)} anchor+neighbor groups")

    return {
        "i_idx": i_idx,            # [E_unique], original edge u-endpoints
        "j_idx": j_idx,            # [E_unique], original edge v-endpoints
        "delta_dataset": delta_dataset,
    }