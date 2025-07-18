import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, dense_to_sparse

class NakamotoConsensus(MessagePassing):
    def __init__(self, num_timesteps=10):
        super().__init__(aggr='max')  # longest chain rule
        self.num_timesteps = num_timesteps
        self.block_counter = 1  

    def forward(self, x, edge_index, hash_power, malicious_mask=None, record_new_tip=False):
        """
        x: Tensor of shape [num_nodes, 2] -> [:, 0] = height, [:, 1] = tip_id
        hash_power: Tensor of shape [num_nodes], values in [0, 1]
        malicious_mask: Bool tensor of shape [num_nodes] indicating malicious nodes
        """
        num_nodes = x.size(0)
        device = x.device
        if malicious_mask is None:
            malicious_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        consensus_over_time = [x.clone().detach()]

        for _ in range(self.num_timesteps):
            heights, tip_ids = x[:, 0], x[:, 1]
            mined = torch.bernoulli(hash_power).int().to(device)

            # handle malicious miners mining fake blocks
            for i in range(num_nodes):
                if mined[i] == 1 and malicious_mask[i]:
                    heights[i] += 1
                    tip_ids[i] = torch.randint(1000, 2000, (1,), device=device).item()

            # honest miners 
            valid_miners = (mined == 1) & (~malicious_mask)
            if valid_miners.any():
                chosen = valid_miners.nonzero().view(-1)[
                    torch.randint(0, valid_miners.sum(), (1,)).item()
                ]
                heights[chosen] += 1
                tip_ids[chosen] = self.block_counter
                self.block_counter += 1

            x = torch.stack([heights, tip_ids], dim=1)
            if record_new_tip:
                consensus_over_time.append(x.clone().detach())

            # propagate longest chain to neighbors
            self.malicious_mask = malicious_mask 
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

        # honest nodes adopt longer chains from neighbors
        honest_mask = ~self.malicious_mask if hasattr(self, 'malicious_mask') else torch.ones_like(current_height, dtype=torch.bool)
        longer = (neighbor_height > current_height) & honest_mask
        new_height[longer] = neighbor_height[longer]
        new_tip[longer] = neighbor_tip[longer]

        # fork detection
        fork_mask = (neighbor_height == current_height) & (neighbor_tip != current_tip)
        self.forks = fork_mask.nonzero(as_tuple=False).view(-1)

        return torch.stack([new_height, new_tip], dim=1)


class ProofOfStakeConsensus(MessagePassing):
    def __init__(self, num_timesteps=10):
        super().__init__(aggr='max')
        self.num_timesteps = num_timesteps
        self.block_counter = 1  
        self.view = 0 
        self.proposer_schedule = None  
        self.malicious_mask = None

    def forward(self, x, edge_index, stake, malicious_mask=None, record_new_tip=False):
        """
        x: Tensor [num_nodes, 2] -> [:, 0] = height, [:, 1] = tip_id
        stake: Tensor [num_nodes] summing to 1 (stake weights)
        malicious_mask: BoolTensor [num_nodes] marking malicious nodes
        """
        num_nodes = x.size(0)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        consensus_over_time = [x.clone().detach()]

        device = x.device
        self.view = 0
        self.malicious_mask = malicious_mask if malicious_mask is not None else torch.zeros(num_nodes, dtype=torch.bool, device=device)

        # precompute proposer schedule based on node stake
        schedule_length = self.num_timesteps * 2
        self.proposer_schedule = torch.multinomial(stake, num_samples=schedule_length, replacement=True)

        for _ in range(self.num_timesteps):
            heights, tip_ids = x[:, 0], x[:, 1]
            proposer = self.proposer_schedule[self.view].item()

            if self.malicious_mask[proposer]:
                # timeout and skip this round and move to the next proposer/view
                self.view += 1
                consensus_over_time.append(x.clone().detach())
                continue

            # honest proposer creates new block
            heights[proposer] += 1
            tip_ids[proposer] = self.block_counter
            self.block_counter += 1
            self.view += 1  # increment view after successful proposal

            x = torch.stack([heights, tip_ids], dim=1)
            if record_new_tip:
                consensus_over_time.append(x.clone().detach())

            # propagate best chain 
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

        honest_mask = ~self.malicious_mask
        longer = (neighbor_height > current_height) & honest_mask
        new_height[longer] = neighbor_height[longer]
        new_tip[longer] = neighbor_tip[longer]

        # fork tracking
        fork_mask = (neighbor_height == current_height) & (neighbor_tip != current_tip)
        self.forks = fork_mask.nonzero(as_tuple=False).view(-1)

        return torch.stack([new_height, new_tip], dim=1)


class PBFTConsensus(MessagePassing):
    def __init__(self, num_timesteps=10, f_max=1):
        """
        f_max: The maximum number of faulty nodes the system can tolerate.
        """
        super().__init__(aggr='add')  # aggregate incoming votes
        self.num_timesteps = num_timesteps
        self.f_max = f_max
        self.block_counter = 1 
        self.view = 0  

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

            # leader determined by the current view proposes new block
            leader = self.view % num_nodes
            
            # malicious leader
            if malicious_mask[leader]:
                # move to the next view/leader in the next timestep if leader is detected as malicious
                self.view += 1
                consensus_over_time.append(x.clone().detach())
                continue
                
            proposed_block_id = self.block_counter
            # leader proposes to extend its own current chain height
            proposed_height = heights[leader] + 1

            valid_proposal_mask = (~malicious_mask) & (proposed_height > heights)

            # prepare phase 
            prepare_votes = torch.zeros(num_nodes, dtype=torch.int, device=x.device)
            prepare_votes[valid_proposal_mask] = 1

            # check prepared state
            total_prepare_votes = self.propagate(full_edge_index, x=prepare_votes.unsqueeze(1)).squeeze(1)
            
            # node is prepared if it received at least 2*f matching prepare messages
            is_prepared_mask = (total_prepare_votes >= 2 * self.f_max) & (~malicious_mask)

            # prepared nodes enter commit phase
            commit_votes = torch.zeros(num_nodes, dtype=torch.int, device=x.device)
            commit_votes[is_prepared_mask] = 1
            
            # nodes check for committed state
            total_commit_votes = self.propagate(full_edge_index, x=commit_votes.unsqueeze(1)).squeeze(1)

            # honest node commits if it receives 2*f + 1 commit messages
            commit_decision_mask = (total_commit_votes >= (2 * self.f_max) + 1) & (~malicious_mask)
            
            # update state for committed nodes 
            if commit_decision_mask.any():
                heights[commit_decision_mask] = proposed_height
                tip_ids[commit_decision_mask] = proposed_block_id
                self.block_counter += 1
                self.view += 1 # move to next view after successful commit
            else:
                self.view += 1

            x = torch.stack([heights, tip_ids], dim=1)
            consensus_over_time.append(x.clone().detach())
        
        return x, consensus_over_time

    def message(self, x_j):
        return x_j