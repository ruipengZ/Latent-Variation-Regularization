import os, sys
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))
from examples.imitation.isaac_env import make_isaac_env
from examples.helpers import knn_epsilon_graph, build_delta_dict

import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt




class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, device):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_sizes[0]),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            torch.nn.ELU(),
        )
        self.projector = torch.nn.Linear(hidden_sizes[2], output_size)
        self.device = device

    def forward(self, x, **kwargs):
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        x = self.features(x)
        output = self.projector(x)
        print('network: ', output)
        return output





def train(input, output, model, delta_dict, optim):
    device = model.device
    i_idx = delta_dict.get("i_idx", None)  # [C, M]
    j_idx = delta_dict.get("j_idx", None)  # [C, M]
    delta_dataset = delta_dict.get("delta_dataset", None)  # [C, M, d_feat]

    delta_batch = random.sample(delta_dataset, args.anchors_per_update)
    anchors, nbrs_lists = zip(*delta_batch)

    L_max = max(args.k_neighbors, max(len(n) for n in nbrs_lists))
    nbrs_tensor = torch.full((args.anchors_per_update, L_max), -1, dtype=torch.long, device=device)
    mask = torch.zeros((args.anchors_per_update, L_max), dtype=torch.bool, device=device)
    for b, nbrs in enumerate(nbrs_lists):
        # sample nbrs up to max_neighbors
        if len(nbrs) > args.k_neighbors:
            nbrs = random.sample(nbrs, args.k_neighbors)
        nbrs_tensor[b, :len(nbrs)] = torch.tensor(nbrs, device=device)
        mask[b, :len(nbrs)] = True

    # Gather delta‑vectors
    input_tensor = torch.from_numpy(input).to(device)
    output_tensor = torch.from_numpy(output).to(device)
    h_all = model.features(input_tensor)

    delta_h_full = h_all[j_idx] - h_all[i_idx]  # [E, d]
    delta_output = output_tensor[j_idx] - output_tensor[i_idx]  # [E, A]

    optim.zero_grad()

    ### === BC loss for the selected anchor delta points
    loss_h_bc = (F.mse_loss(model.projector(h_all[i_idx]), output_tensor[i_idx]) +
                 F.mse_loss(model.projector(h_all[j_idx]), output_tensor[j_idx])) / 2

    ## === project delta_h_full to row space
    W = model.projector.weight  # [d_a, d_h]
    WWt_inv = torch.inverse(W @ W.T)  # [d_a, d_a]
    P_null = torch.eye(W.size(-1), device=W.device) - W.T @ WWt_inv @ W  # [d_h, d_h]

    delta_h_null = (P_null @ delta_h_full.T).T
    delta_h_row = delta_h_full - delta_h_null

    ### ===== KL loss
    loss_kl_row, H_h_row = kl_loss(delta_h_row, delta_output, anchors, nbrs_tensor, args.tau, mask)
    loss_kl = loss_kl_row

    # loss_kl = torch.tensor(0.0, device=device)

    loss = loss_h_bc + args.lambda_kl * loss_kl


    loss.backward()
    optim.step()

    result = {
        "loss": loss.item(),
        "loss_h_bc": loss_h_bc.item(),
        "loss_kl": loss_kl.item(),
    }

    return result


def kl_loss(delta_h, delta_output, anchors, nbrs_tensor, tau, mask, eps=1e-8):

    dh_anchor = delta_h[list(anchors)]  # [B, d]
    da_anchor = delta_output[list(anchors)]  # [B, A]
    dh_nbr = delta_h[nbrs_tensor.clamp(min=0)]  # [B, L, d]
    da_nbr = delta_output[nbrs_tensor.clamp(min=0)]  # [B, L, A]

    # Compute per‐anchor cosine sims (vectorized)
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

    loss_kl = kl.mean()
    return loss_kl, H_h





if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description="Training / evaluation config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -------------------------
    # Model
    # -------------------------
    p.add_argument("--input-size", type=int, default=48)
    p.add_argument("--output-size", type=int, default=12)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128, 128])  # e.g. 128 128 128

    # -------------------------
    # Data
    # -------------------------
    p.add_argument("--task", type=str, default='Isaac-Velocity-Flat-Forward-Unitree-Go2-v0')
    p.add_argument("--dataset-path", type=str, default='imitation_data/isaac_go2_forward/traj-1.hdf5')
    p.add_argument("--output-dir", type=str, default='results/minimal_LVR')

    # -------------------------
    # Training
    # -------------------------
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-epochs", type=int, default=15000)

    # -------------------------
    #Hyperparameters
    # -------------------------
    p.add_argument("--k-neighbors", type=int, default=32)
    p.add_argument("--anchors-per-update", type=int, default=32)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--lambda-kl", type=float, default=0.1)

    # -------------------------
    # Runtime
    # -------------------------
    p.add_argument("--headless", type=int, default=True)
    p.add_argument("--training", type=int, default=True)
    p.add_argument("--testing", type=int, default=True)

    args = p.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(args.input_size, args.hidden_sizes, args.output_size, args.device).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.training:
        ### buffer
        from tianshou.data.buffer.base import ReplayBuffer
        buffer = ReplayBuffer.load_hdf5(args.dataset_path)
        input = buffer.obs[:len(buffer)]  # [N, D]
        output = buffer.act[:len(buffer)]

        X = torch.from_numpy(input)
        edge_index = knn_epsilon_graph(X, k=args.k_neighbors, metric="euclidean", q=1)
        delta_dict = build_delta_dict(edge_index,
                                      num_nodes=len(X),
                                      device=None,
                                      min_neighbors=None,
                                      max_neighbors=None,
                                      )

        model.train()

        # Turn on interactive mode
        plt.ion()
        fig, ax = plt.subplots(1,1,figsize=(10,5))

        line_bc, = ax.plot([], [], label="BC Loss")
        line_kl, = ax.plot([], [], label="KL Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.set_xlim(-100, args.n_epochs)
        ax.set_ylim(0, 2.3)

        plt.show()


        epochs = []
        losses_bc = []
        losses_kl = []


        for epoch in range(args.n_epochs):
            train_result = train(input, output, model, delta_dict, optim)
            loss_bc = float(train_result["loss_h_bc"])
            loss_kl = float(train_result["loss_kl"])

            print(f"\rEpoch {epoch:5d} | BC loss: {loss_bc:.4f} | KL loss: {loss_kl:.4f}", end="", flush=True)

            # Update data
            epochs.append(epoch)
            losses_bc.append(loss_bc)
            losses_kl.append(loss_kl)

            # Update plot
            line_bc.set_xdata(epochs)
            line_bc.set_ydata(losses_bc)
            line_kl.set_xdata(epochs)
            line_kl.set_ydata(losses_kl)

            plt.pause(0.001)  # update GUI

        print("\n======= Training Finished !! =======")  # newline after loop ends
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{args.output_dir}/policy.pt")
        plt.savefig(f"{args.output_dir}/loss.png")


    ## ====================== for testing ========================== ####
    if args.testing:
        from tianshou.data.isaac_collector import Collector
        from tianshou.policy.imitation.base import ImitationPolicy
        from tianshou.utils.net.continuous import UnboundedActor
        from tianshou.utils.net.common import Net
        import pprint


        # model
        net = Net(
            (args.input_size,),
            hidden_sizes=args.hidden_sizes,
            activation=torch.nn.ELU,
            device=args.device,
        )

        actor = UnboundedActor(
            net,
            (args.output_size,),
            device=args.device,
        ).to(args.device)


        model_state_dict = torch.load(f"{args.output_dir}/policy.pt")
        # some renames of weights (compatible with tianshou)
        new_state_features = {}
        new_state_projector = {}
        for k, v in model_state_dict.items():
            if "features" in k:
                new_state_features[k.replace("features", "model.model")] = v
            elif "projector" in k:
                new_state_projector[k.replace("projector", "model.0")] = v
        net.load_state_dict(new_state_features)
        actor.last.load_state_dict(new_state_projector)


        optim = torch.optim.AdamW(net.parameters(), lr=args.lr)


        env, test_envs, simulation_app = make_isaac_env(
            args.task, 1, 1000, obs_norm=False, headless=args.headless,
        )


        # define policy
        policy = ImitationPolicy(
            actor,
            optim,
            action_space=env.action_space,
        ).to(args.device)



        test_collector = Collector(policy, test_envs)

        print("Setup test envs ...")
        policy.eval()
        test_envs.seed(0)
        print("Testing agent ...")
        test_collector.reset()
        final_result = test_collector.collect(n_episode=1000, render=0.)
        pprint.pprint(final_result)
        rew = final_result["rews"].mean()
        print(f'Mean reward (over { final_result["n/ep"]} episodes): {rew}')


        env.close()
        simulation_app.close()