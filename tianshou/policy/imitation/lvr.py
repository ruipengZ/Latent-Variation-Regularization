import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, to_torch
from tianshou.policy import BasePolicy



class LVRImitationPolicy(BasePolicy):

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        num_anchors: int,
        max_neighbors: int,
        tau: float,
        lambda_kl: float,
        cosine_threshold: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        assert self.action_type in ["continuous", "discrete"], \
            "Please specify action_space."

        self.num_anchors = num_anchors
        self.max_neighbors = max_neighbors
        self.tau = tau
        self.lambda_kl = lambda_kl
        self.cosine_threshold = cosine_threshold

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        logits, hidden = self.model(batch.obs, state=state, info=batch.info)
        if self.action_type == "discrete":
            act = logits.max(dim=1)[1]
        else:
            act = logits
        return Batch(logits=logits, act=act, state=hidden)

    def update(self, sample_size: int, buffer, delta_dict,
               **kwargs: Any) -> Dict[str, Any]:

        if buffer is None:
            return {}

        batch, indices = buffer.sample(0)
        self.updating = True
        batch = self.process_fn(batch, buffer, indices)

        i_idx = delta_dict.get("i_idx", None)  # [C, M]
        j_idx = delta_dict.get("j_idx", None)  # [C, M]
        delta_dataset = delta_dict.get("delta_dataset", None)  # [C, M, d_feat]

        ## Sample a batch of local anchors
        delta_batch = random.sample(delta_dataset, self.num_anchors)
        anchors, nbrs_lists = zip(*delta_batch)

        # 4) Turn to tensors (pad neighbor lists to L_max):
        L_max = max(self.max_neighbors, max(len(n) for n in nbrs_lists))
        nbrs_tensor = torch.full((self.num_anchors, L_max), -1, dtype=torch.long, device=self.model.device)
        mask = torch.zeros((self.num_anchors, L_max), dtype=torch.bool, device=self.model.device)
        for b, nbrs in enumerate(nbrs_lists):
            # sample nbrs up to max_neighbors
            if len(nbrs) > self.max_neighbors:
                nbrs = random.sample(nbrs, self.max_neighbors)
            nbrs_tensor[b, :len(nbrs)] = torch.tensor(nbrs, device=self.model.device)
            mask[b, :len(nbrs)] = True

        # 5) Gather delta‑vectors
        h_all = self.model.preprocess(buffer.obs[:len(buffer)])[0]  # [B, d]
        a_gt_all = torch.from_numpy(buffer.act[:len(buffer)]).to(self.model.device)
        delta_h_full = h_all[j_idx] - h_all[i_idx]  # [E, d]
        delta_a_gt = a_gt_all[j_idx] - a_gt_all[i_idx]  # [E, A]

        result = self.learn(batch, buffer, i_idx, j_idx,
                            h_all, a_gt_all,
                            delta_h_full, delta_a_gt,
                            anchors, nbrs_tensor, mask,
                            **kwargs)

        self.post_process_fn(batch, buffer, indices)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False
        return result


    def kl_loss(self, delta_h, delta_a, anchors, nbrs_tensor, tau, mask, eps=1e-8):

        dh_anchor = delta_h[list(anchors)]  # [B, d]
        da_anchor = delta_a[list(anchors)]  # [B, A]
        dh_nbr = delta_h[nbrs_tensor.clamp(min=0)]  # [B, L, d]
        da_nbr = delta_a[nbrs_tensor.clamp(min=0)]  # [B, L, A]

        t0 = self.cosine_threshold  # cosine threshold
        cos_a_gate = F.cosine_similarity(da_anchor.unsqueeze(1), da_nbr, dim=2)  # [B, L]
        pos_mass = ((abs(cos_a_gate) - t0).clamp_min(0) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        lambda_dist = pos_mass.clamp(0, 1).detach()

        cos_h = F.cosine_similarity(dh_anchor.unsqueeze(1), dh_nbr, dim=2)  # [B, L]
        cos_a = F.cosine_similarity(da_anchor.unsqueeze(1), da_nbr, dim=2)  # [B, L]

        def masked_softmax(x, m):
            ex = torch.exp(x) * m.float()
            s = ex.sum(dim=1, keepdim=True).clamp(min=eps)
            return ex / s

        p_h = masked_softmax(cos_h / tau, mask)  # [B, L]
        p_a = masked_softmax(cos_a / tau, mask)  # [B, L]

        kl = (p_a * (torch.log(p_a + eps) - torch.log(p_h + eps))).sum(dim=1)

        H_h = -(p_h * (p_h.clamp_min(eps)).log()).sum(1)  # per-row entropy

        loss_kl = (lambda_dist * kl).mean()
        return loss_kl, H_h



    def learn(self, batch: Batch, buffer,
              i_idx, j_idx,
              h_all, a_gt_all,
              delta_h_full, delta_a_gt,
              anchors, nbrs_tensor, mask,
              **kwargs: Any) -> Dict[str, float]:

        self.optim.zero_grad()

        ### === BC loss for the selected anchor delta point (already done forward pass in def update)
        loss_bc = (F.mse_loss(self.model.last.model[-1](h_all[i_idx]), a_gt_all[i_idx]) +
                     F.mse_loss(self.model.last.model[-1](h_all[j_idx]), a_gt_all[j_idx]))/2

        ### === project delta_h_full to row space
        W = self.model.last.model[-1].weight  # [d_a, d_h]

        WWt_inv = torch.inverse(W @ W.T)  # [d_a, d_a]
        P_null = torch.eye(W.size(-1), device=W.device) - W.T @ WWt_inv @ W  # [d_h, d_h]

        delta_h_null = (P_null @ delta_h_full.T).T
        delta_h_row = delta_h_full - delta_h_null

        ### ===== KL loss
        loss_kl, H_h_row = self.kl_loss(delta_h_row, delta_a_gt, anchors, nbrs_tensor, self.tau, mask)

        loss = loss_bc + self.lambda_kl * loss_kl


        loss.backward()
        self.optim.step()
        return {
            "loss": loss.item(),
            "loss_bc": loss_bc.item(),
            "loss_kl": loss_kl.item(),

            "delta_h_null": (delta_h_null.norm(dim=1)).mean().item(),
            "delta_h_row": (delta_h_row.norm(dim=1)).mean().item(),
            "delta_h": (delta_h_full.norm(dim=1)).mean().item(),

            "entropy_row": H_h_row.mean().item(),

        }



