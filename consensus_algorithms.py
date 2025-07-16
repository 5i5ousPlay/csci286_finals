import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class NakamotoConsensus(MessagePassing):
    def __init__(self, num_timesteps=10):
        super().__init__(aggr='max')  # max height propagation
        self.num_timesteps = num_timesteps
        self.block_counter = 1  # start assigning block IDs from 1 (0 = genesis)

    def forward(self, x, edge_index, hash_power, record_new_tip=False):
        """
        x: Tensor of shape [num_nodes, 2] -> [:, 0] = height, [:, 1] = tip_id
        hash_power: Tensor of shape [num_nodes], values in [0, 1]
        """
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        consensus_over_time = [x.clone().detach()]
        for _ in range(self.num_timesteps):
            heights, tip_ids = x[:, 0], x[:, 1]

            # mining, only select nodes successfully mine
            mined = torch.bernoulli(hash_power).int().to(x.device)
            for i in range(x.size(0)):
                if mined[i] == 1:
                    heights[i] += 1
                    tip_ids[i] = self.block_counter
                    self.block_counter += 1

            x = torch.stack([heights, tip_ids], dim=1)
            if record_new_tip:
                consensus_over_time.append(x.clone().detach())

            # propagate best chain info from neighbors
            x = self.propagate(edge_index, x=x, current_x=x)

            consensus_over_time.append(x.clone().detach())
        return x, consensus_over_time

    def message(self, x_j):
        # Send neighbor’s height and tip ID
        return x_j

    def update(self, aggr_out, current_x):
        current_height, current_tip = current_x[:, 0], current_x[:, 1]
        neighbor_height, neighbor_tip = aggr_out[:, 0], aggr_out[:, 1]

        new_height = current_height.clone()
        new_tip = current_tip.clone()

        # if neighbor has longer chain, adopt it
        longer = neighbor_height > current_height
        new_height[longer] = neighbor_height[longer]
        new_tip[longer] = neighbor_tip[longer]

        # track where heights match but tips differ
        fork_mask = (neighbor_height == current_height) & (neighbor_tip != current_tip)
        self.forks = fork_mask.nonzero(as_tuple=False).view(-1)

        return torch.stack([new_height, new_tip], dim=1)


class ProofOfStakeConsensus(MessagePassing):
    def __init__(self, num_timesteps=10):
        super().__init__(aggr='max')
        self.num_timesteps = num_timesteps
        self.block_counter = 1  # start assigning block IDs from 1 (0 = genesis)

    def forward(self, x, edge_index, stake, record_new_tip=False):
        """
        x: Tensor of shape [num_nodes, 2] -> [:, 0] = height, [:, 1] = tip_id
        stake: Tensor of shape [num_nodes], values in [0, 1], summing to 1
        """
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        consensus_over_time = [x.clone().detach()]

        for _ in range(self.num_timesteps):
            heights, tip_ids = x[:, 0], x[:, 1]

            # --- Step 1: Select block proposer based on stake ---
            proposer = torch.multinomial(stake, 1).item()
            # print(proposer)
            heights[proposer] += 1
            tip_ids[proposer] = self.block_counter
            self.block_counter += 1

            x = torch.stack([heights, tip_ids], dim=1)
            if record_new_tip:
                consensus_over_time.append(x.clone().detach())

            # --- Step 2: Propagate best chain ---
            x = self.propagate(edge_index, x=x, current_x=x)
            consensus_over_time.append(x.clone().detach())

        return x, consensus_over_time

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, current_x):
        current_height, current_tip = current_x[:, 0], current_x[:, 1]
        neighbor_height, neighbor_tip = aggr_out[:, 0], aggr_out[:, 1]

        new_height = current_height.clone()
        new_tip = current_tip.clone()

        longer = neighbor_height > current_height
        new_height[longer] = neighbor_height[longer]
        new_tip[longer] = neighbor_tip[longer]

        # Optional fork tracking
        fork_mask = (neighbor_height == current_height) & (neighbor_tip != current_tip)
        self.forks = fork_mask.nonzero(as_tuple=False).view(-1)

        return torch.stack([new_height, new_tip], dim=1)