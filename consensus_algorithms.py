import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class NakamotoConsensus(MessagePassing):
    def __init__(self, num_timesteps=10):
        super().__init__(aggr='max')  # max height propagation
        self.num_timesteps = num_timesteps
        self.block_counter = 1  # start assigning block IDs from 1 (0 = genesis)

    def forward(self, x, edge_index, hash_power, malicious_mask=None, record_new_tip=False):
        """
        x: Tensor of shape [num_nodes, 2] -> [:, 0] = height, [:, 1] = tip_id
        hash_power: Tensor of shape [num_nodes], values in [0, 1]
        malicious_mask: Bool tensor of shape [num_nodes] indicating malicious nodes
        """
        if malicious_mask is None:
            malicious_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        consensus_over_time = [x.clone().detach()]

        for _ in range(self.num_timesteps):
            heights, tip_ids = x[:, 0], x[:, 1]
            mined = torch.bernoulli(hash_power).int().to(x.device)

            for i in range(x.size(0)):
                if mined[i] == 1:
                    if malicious_mask[i]:
                        # Malicious nodes mine a block with fake tip ID
                        heights[i] += 1
                        tip_ids[i] = torch.randint(1000, 2000, (1,), device=x.device).item()
                    else:
                        heights[i] += 1
                        tip_ids[i] = self.block_counter
                        self.block_counter += 1

            x = torch.stack([heights, tip_ids], dim=1)
            if record_new_tip:
                consensus_over_time.append(x.clone().detach())

            # Propagate best chain from neighbors
            self.malicious_mask = malicious_mask  # store for use in `update`
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

        # Honest nodes adopt longer chains from neighbors
        if hasattr(self, 'malicious_mask'):
            honest_mask = ~self.malicious_mask
        else:
            honest_mask = torch.ones_like(current_height, dtype=torch.bool)

        longer = (neighbor_height > current_height) & honest_mask
        new_height[longer] = neighbor_height[longer]
        new_tip[longer] = neighbor_tip[longer]

        # Fork detection (optional, all nodes)
        fork_mask = (neighbor_height == current_height) & (neighbor_tip != current_tip)
        self.forks = fork_mask.nonzero(as_tuple=False).view(-1)

        return torch.stack([new_height, new_tip], dim=1)


class ProofOfStakeConsensus(MessagePassing):
    def __init__(self, num_timesteps=10):
        super().__init__(aggr='max')
        self.num_timesteps = num_timesteps
        self.block_counter = 1  # start assigning block IDs from 1 (0 = genesis)
        self.malicious_mask = None  # to be set externally or via forward

    def forward(self, x, edge_index, stake, malicious_mask=None, record_new_tip=False):
        """
        x: Tensor of shape [num_nodes, 2] -> [:, 0] = height, [:, 1] = tip_id
        stake: Tensor of shape [num_nodes], values in [0, 1], summing to 1
        malicious_mask: BoolTensor of shape [num_nodes] indicating malicious nodes (True = malicious)
        """
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        consensus_over_time = [x.clone().detach()]
        self.malicious_mask = malicious_mask if malicious_mask is not None else torch.zeros(x.size(0), dtype=torch.bool, device=x.device)

        for _ in range(self.num_timesteps):
            heights, tip_ids = x[:, 0], x[:, 1]

            # --- Step 1: Select block proposer based on stake ---
            proposer = torch.multinomial(stake, 1).item()
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

        if self.malicious_mask is None:
            self.malicious_mask = torch.zeros_like(current_height, dtype=torch.bool)

        # Honest nodes adopt longer chains
        honest_mask = ~self.malicious_mask
        longer = (neighbor_height > current_height) & honest_mask
        new_height[longer] = neighbor_height[longer]
        new_tip[longer] = neighbor_tip[longer]

        # Optional fork tracking
        fork_mask = (neighbor_height == current_height) & (neighbor_tip != current_tip)
        self.forks = fork_mask.nonzero(as_tuple=False).view(-1)

        return torch.stack([new_height, new_tip], dim=1)


class PBFTConsensus(MessagePassing):
    def __init__(self, num_timesteps=10, f_max=1):
        """
        f_max: The maximum number of faulty nodes the system can tolerate.
        """
        super().__init__(aggr='add')  # Use 'add' to sum incoming votes
        self.num_timesteps = num_timesteps
        self.f_max = f_max
        self.block_counter = 1  # Start assigning block IDs from 1
        self.view = 0  # Initial view number

    def forward(self, x, edge_index, malicious_mask=None, record_new_tip=False):
        """
        x: Tensor of shape [num_nodes, 2] -> [:, 0] = height, [:, 1] = tip_id
        malicious_mask: Boolean tensor, True for malicious nodes.
        """
        num_nodes = x.size(0)

        full_adj = torch.ones(num_nodes, num_nodes, device=x.device)
        full_edge_index, _ = dense_to_sparse(full_adj)


        if malicious_mask is None:
            malicious_mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)

        consensus_over_time = [x.clone().detach()]

        for t in range(self.num_timesteps):
            heights, tip_ids = x[:, 0], x[:, 1]

            # --- Step 1: Leader proposes a new block (Pre-prepare Phase) ---
            # Leader is determined by the current view
            leader = self.view % num_nodes
            
            # If the leader is malicious, it might not propose anything.
            if malicious_mask[leader]:
                # move to the next view/leader in the next timestep if leader is detected as malicious
                self.view += 1
                consensus_over_time.append(x.clone().detach())
                continue
                
            proposed_block_id = self.block_counter
            # The leader proposes to extend its own current chain height.
            proposed_height = heights[leader] + 1

            valid_proposal_mask = (~malicious_mask) & (proposed_height > heights)

            # --- Step 2: Honest nodes enter the Prepare Phase ---
            prepare_votes = torch.zeros(num_nodes, dtype=torch.int, device=x.device)
            prepare_votes[valid_proposal_mask] = 1

            # --- Step 3: Nodes check for "Prepared" state ---
            total_prepare_votes = self.propagate(full_edge_index, x=prepare_votes.unsqueeze(1)).squeeze(1)
            
            # A node is "prepared" if it received at least 2*f matching prepare messages.
            is_prepared_mask = (total_prepare_votes >= 2 * self.f_max) & (~malicious_mask)

            # --- Step 4: "Prepared" nodes enter the Commit Phase ---
            commit_votes = torch.zeros(num_nodes, dtype=torch.int, device=x.device)
            commit_votes[is_prepared_mask] = 1
            
            # --- Step 5: Nodes check for "Committed" state ---
            total_commit_votes = self.propagate(full_edge_index, x=commit_votes.unsqueeze(1)).squeeze(1)

            # An honest node commits if it receives 2*f + 1 commit messages.
            commit_decision_mask = (total_commit_votes >= (2 * self.f_max) + 1) & (~malicious_mask)
            
            # --- Step 6: Update State for Committed Nodes ---
            if commit_decision_mask.any():
                heights[commit_decision_mask] = proposed_height
                tip_ids[commit_decision_mask] = proposed_block_id
                self.block_counter += 1
                self.view += 1 # Move to the next view after a successful commit
            else:
                self.view += 1

            x = torch.stack([heights, tip_ids], dim=1)
            consensus_over_time.append(x.clone().detach())
        
        return x, consensus_over_time

    def message(self, x_j):
        return x_j