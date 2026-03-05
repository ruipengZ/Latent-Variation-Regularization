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


@torch.no_grad()
def inspect_neighbors(x, edge_index, nodes=None, undirected=True, metric="euclidean", top=None):

    s, d = edge_index
    E = edge_index.size(1)
    N = len(x)
    deg = degree(s, num_nodes=N)

    print(f"[kNN-ε] N={N:,}  E={E:,}  avg_deg={deg.mean().item():.2f}  "
          f"med_deg={deg.median().item():.0f}  min/max_deg=({int(deg.min())}/{int(deg.max())})")

    print(f"[kNN-ε] isolated_nodes={int((deg == 0).sum())}")
    iso_idx = (deg == 0).nonzero(as_tuple=True)[0]
    print(f"[kNN-ε] isolated idx: {iso_idx.tolist()}")

    # fraction of edges whose reverse exists (how symmetric/mutual the graph is)
    rev_frac = torch.isin(s * N + d, d * N + s).float().mean().item()
    print(f"[kNN-ε] reciprocal_edge_fraction={rev_frac:.3f}")

    N = x.size(0)
    s, d = edge_index
    if not undirected:
        # union in- and out-neighbors
        s = torch.cat([s, d]); d = torch.cat([d, s])

    # distances for each (s->d)
    if metric == "euclidean":
        dist = (x[s] - x[d]).pow(2).sum(-1).sqrt()
    elif metric == "cosine":
        xn = F.normalize(x, dim=-1)
        dist = 1.0 - (xn[s] * xn[d]).sum(-1)
    else:
        raise ValueError("metric must be 'euclidean' or 'cosine'")

    # group neighbors by source
    idx = torch.argsort(s)                       # group edges by src
    s, d, dist = s[idx], d[idx], dist[idx]
    counts = torch.bincount(s, minlength=N)      # degree per node
    ptr = torch.empty(N+1, dtype=torch.long, device=x.device)
    ptr[0] = 0
    ptr[1:] = counts.cumsum(0)

    # choose which nodes to print
    if nodes is None:
        noniso = torch.where(counts > 0)[0]
        iso    = torch.where(counts == 0)[0]
        picks = []
        if noniso.numel():
            # evenly sample up to 5 non-isolated nodes
            step = max(1, noniso.numel() // 5)
            picks += noniso[::step][:5].tolist()
        if iso.numel():
            picks += iso[:min(3, iso.numel())].tolist()
        nodes = picks or [0]

    # print neighbors
    for i in nodes:
        a, b = ptr[i].item(), ptr[i+1].item()
        nbrs = d[a:b]; nd = dist[a:b]
        if nbrs.numel():
            order = torch.argsort(nd)
            nbrs, nd = nbrs[order], nd[order]
            if top is not None:
                nbrs, nd = nbrs[:top], nd[:top]
            print(f"[nbrs] i={i}  deg={nbrs.numel()}  "
                  f"neighbors={nbrs.tolist()}  dists={[round(x,4) for x in nd.tolist()]}")
        else:
            print(f"[nbrs] i={i}  deg=0  neighbors=[]")


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