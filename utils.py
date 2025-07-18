import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from tqdm import tqdm
from IPython.display import HTML
from matplotlib import animation

_CONVERGENCE_CONTEXT = {}

def count_forks(consensus_over_time):
    """
    Counts the number of fork events across all timesteps.
    A fork at a given timestep occurs when there are multiple tip IDs at the same height.
    
    Returns:
        total_fork_events: int
        forks_per_step: List[int]
    """
    total_fork_events = 0
    forks_per_step = []

    for state in consensus_over_time:
        heights = state[:, 0]
        tips = state[:, 1]
        forks_this_step = 0

        unique_heights = torch.unique(heights)

        for h in unique_heights:
            mask = heights == h
            tips_at_h = tips[mask]
            if len(torch.unique(tips_at_h)) > 1:
                forks_this_step += 1

        forks_per_step.append(forks_this_step)
        total_fork_events += forks_this_step

    return total_fork_events, forks_per_step


def compute_convergence_time(state_over_time, threshold=0.9):
    """
    Returns the first timestep when `threshold` fraction of HONEST nodes agree on the same tip_id.
    If never reached, returns -1.
    """
    malicious_mask = _CONVERGENCE_CONTEXT.get("malicious_mask", None)

    for t, state in enumerate(state_over_time):
        tip_ids = state[:, 1].cpu().numpy()

        if malicious_mask is not None:
            honest_mask = ~malicious_mask.cpu().numpy()
            tip_ids = tip_ids[honest_mask]

        tip_counts = Counter(tip_ids)
        most_common_count = max(tip_counts.values())
        frac_agree = most_common_count / len(tip_ids)

        if frac_agree >= threshold:
            return t

    return -1
    # """
    # Returns the first timestep when `threshold` fraction of nodes agree on the same tip_id.
    # If never reached, returns -1.
    # """
    # for t, state in enumerate(state_over_time):
    #     tip_ids = state[:, 1].cpu().numpy()
    #     tip_counts = Counter(tip_ids)
    #     most_common_count = max(tip_counts.values())
    #     frac_agree = most_common_count / len(tip_ids)

    #     if frac_agree >= threshold:
    #         return t  # Convergence time
    
    # return -1  # No convergence within the simulation window


def compute_throughput(history, num_timesteps):
    committed_blocks = len(torch.unique(history[-1][:, 1])) - 1
    throughput = committed_blocks / num_timesteps
    return throughput


def gini_coefficient_tip_distribution(state):
    tips = state[:, 1].tolist()
    tip_counts = np.array([tips.count(t) for t in set(tips)])
    tip_counts_sorted = np.sort(tip_counts)
    n = len(tip_counts_sorted)
    cumulative = np.cumsum(tip_counts_sorted)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return gini


def average_gini_over_time(states):
    ginis = [gini_coefficient_tip_distribution(state) for state in states]
    return np.mean(ginis)


def compute_tip_fluctuation(consensus_over_time):
    agreement_scores = []

    for state in consensus_over_time:
        tip_ids = state[:, 1].cpu().numpy()
        tip_counts = Counter(tip_ids)
        most_common = max(tip_counts.values())
        frac_agree = most_common / len(tip_ids)
        agreement_scores.append(frac_agree)

    return np.mean(agreement_scores)


def monte_carlo_fork(model, *model_args, iterations=100):
    model_forks = []

    for iteration in tqdm(range(iterations), desc="Simulating Network"):
        final_state, state_over_time = model(*model_args)
        total_forks, forks_each_step = count_forks(state_over_time)
        model_forks.append(total_forks)

    out = np.array(model_forks)

    results = {
        "forks": out,
        "total": int(out.sum()),
        "mean": out.mean(),
        "median": np.median(out),
        "std": np.std(out),
        "max": out.max(),
        "min": out.min(),
    }

    return results


def monte_carlo_convergence(model, *model_args, iterations=100, threshold=0.9):
    convergence_times = []

    for _ in tqdm(range(iterations), desc="Simulating Convergence"):
        _, state_over_time = model(*model_args)
        convergence_time = compute_convergence_time(state_over_time, threshold)
        convergence_times.append(convergence_time)

    times = np.array(convergence_times)

    return {
        "convergence_times": times,
        "mean": times[times >= 0].mean() if np.any(times >= 0) else -1,
        "median": np.median(times[times >= 0]) if np.any(times >= 0) else -1,
        "std": np.std(times[times >= 0]) if np.any(times >= 0) else -1,
        "max": times.max(),
        "min": times[times >= 0].min() if np.any(times >= 0) else -1,
        "converged_fraction": np.mean(times >= 0)
    }


def monte_carlo_throughput(model, num_timesteps, *model_args, iterations=100):
    throughputs = []
    for _ in tqdm(range(iterations), desc="Simulating Network"):
        _, history = model(*model_args)
        throughput = compute_throughput(history, num_timesteps)
        throughputs.append(throughput)

    out = np.array(throughputs)
    
    results = {
        "forks": out,
        "total": int(out.sum()),
        "mean": out.mean(),
        "median": np.median(out),
        "std": np.std(out),
        "max": out.max(),
        "min": out.min(),
    }

    return results


def monte_carlo_gini(model, *model_args, iterations=100):
    gini_scores = []

    for _ in tqdm(range(iterations), desc="Simulating Gini"):
        _, state_over_time = model(*model_args)
        gini = average_gini_over_time(state_over_time)
        gini_scores.append(gini)

    gini_scores = np.array(gini_scores)
    return {
        "mean": gini_scores.mean(),
        "median": np.median(gini_scores),
        "std": np.std(gini_scores),
        "max": gini_scores.max(),
        "min": gini_scores.min()
    }


def monte_carlo_tf(model, *model_args, iterations=100):
    tf_scores = []

    for _ in tqdm(range(iterations), desc="Simulating Network"):
        _, state_over_time = model(*model_args)
        tf = compute_tip_fluctuation(state_over_time)
        tf_scores.append(tf)

    out = np.array(tf_scores)

    return {
        "mean": out.mean(),
        "median": np.median(out),
        "std": np.std(out),
        "max": out.max(),
        "min": out.min(),
    }


def unified_monte_carlo_analysis(model, num_timesteps, *model_args, iterations=100, threshold=0.9):
    convergence_times = []
    fork_counts = []
    throughputs = []
    gini_scores = []
    tip_fluctuations = []

    try:
        for arg in model_args:
            if isinstance(arg, torch.Tensor) and arg.dtype == torch.bool:
                _CONVERGENCE_CONTEXT["malicious_mask"] = arg
                break
    except Exception:
        _CONVERGENCE_CONTEXT["malicious_mask"] = None

    for _ in tqdm(range(iterations), desc="Unified Monte Carlo Simulation"):
        final_state, state_over_time = model(*model_args)

        # Convergence
        convergence_time = compute_convergence_time(state_over_time, threshold)
        convergence_times.append(convergence_time)

        # Forks
        total_forks, _ = count_forks(state_over_time)
        fork_counts.append(total_forks)

        # Throughput
        throughput = compute_throughput(state_over_time, num_timesteps)
        throughputs.append(throughput)

        # Gini Coefficient
        gini = average_gini_over_time(state_over_time)
        gini_scores.append(gini)

        # Tip Fluctuation
        tf = compute_tip_fluctuation(state_over_time)
        tip_fluctuations.append(tf)

    # Convert to arrays
    convergence_times = np.array(convergence_times)
    fork_counts = np.array(fork_counts)
    throughputs = np.array(throughputs)
    gini_scores = np.array(gini_scores)
    tip_fluctuations = np.array(tip_fluctuations)

    return {
        "convergence": {
            "mean": convergence_times[convergence_times >= 0].mean() if np.any(convergence_times >= 0) else -1,
            "median": np.median(convergence_times[convergence_times >= 0]) if np.any(convergence_times >= 0) else -1,
            "std": np.std(convergence_times[convergence_times >= 0]) if np.any(convergence_times >= 0) else -1,
            "max": convergence_times.max(),
            "min": convergence_times[convergence_times >= 0].min() if np.any(convergence_times >= 0) else -1,
            "converged_fraction": np.mean(convergence_times >= 0),
        },
        "forks": {
            "mean": fork_counts.mean(),
            "median": np.median(fork_counts),
            "std": np.std(fork_counts),
            "max": fork_counts.max(),
            "min": fork_counts.min(),
        },
        "throughput": {
            "mean": throughputs.mean(),
            "median": np.median(throughputs),
            "std": np.std(throughputs),
            "max": throughputs.max(),
            "min": throughputs.min(),
        },
        "gini": {
            "mean": gini_scores.mean(),
            "median": np.median(gini_scores),
            "std": np.std(gini_scores),
            "max": gini_scores.max(),
            "min": gini_scores.min(),
        },
        "agreement_ratio": {
            "mean": tip_fluctuations.mean(),
            "median": np.median(tip_fluctuations),
            "std": np.std(tip_fluctuations),
            "max": tip_fluctuations.max(),
            "min": tip_fluctuations.min(),
        }
    }



def animate_consensus_networkx(consensus_over_time, edge_index, interval=500, save_to_file=False, filename='graph_animation'):
    """
    Visualizes consensus evolution using NetworkX graphs.
    
    Parameters:
    - consensus_over_time: list of tensors [num_nodes, 2]
    - edge_index: torch.LongTensor of shape [2, num_edges]
    - interval: animation frame interval (ms)
    """
    num_steps = len(consensus_over_time)
    num_nodes = consensus_over_time[0].size(0)

    # 1. Build NetworkX graph from edge_index
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)

    # 2. Assign fixed layout
    pos = nx.spring_layout(G, seed=42)  # or kamada_kawai_layout / circular_layout

    # 3. Tip ID color map
    all_tips = torch.cat([state[:, 1] for state in consensus_over_time])
    unique_tips = torch.unique(all_tips).tolist()
    tip_to_color = {tip_id: plt.cm.tab20(i % 20) for i, tip_id in enumerate(unique_tips)}

    # 4. Initialize figure
    fig, ax = plt.subplots(figsize=(7, 5))

    def update(frame):
        ax.clear()
        state = consensus_over_time[frame]
        heights = state[:, 0].tolist()
        tips = state[:, 1].tolist()
        node_colors = [tip_to_color[tip] for tip in tips]
        labels = {i: f"{int(heights[i])}" for i in range(num_nodes)}

        nx.draw(
            G, pos, ax=ax,
            node_color=node_colors,
            with_labels=True,
            labels=labels,
            node_size=300,
            font_size=8,
            font_color="white",
            edge_color="gray"
        )
        ax.set_title(f"Timestep {frame}")
        ax.axis("off")

    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=interval, repeat=False)
    if save_to_file:
        ani.save(f'{filename}.gif', writer='pillow', fps=2)
    plt.close(fig)
    return HTML(ani.to_jshtml())


def animate_consensus_evolution(consensus_over_time, interval=500):
    """
    Visualizes the evolution of chain height and tip ID over time.
    - consensus_over_time: list of tensors of shape [num_nodes, 2]
    - interval: delay between frames in milliseconds
    """
    num_steps = len(consensus_over_time)
    num_nodes = consensus_over_time[0].size(0)

    # Get all unique tip IDs for consistent coloring
    all_tips = torch.cat([state[:, 1] for state in consensus_over_time])
    unique_tips = torch.unique(all_tips).tolist()
    tip_to_color = {tip_id: plt.cm.tab20(i % 20) for i, tip_id in enumerate(unique_tips)}

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(num_nodes), consensus_over_time[0][:, 0].tolist(), color='gray')

    def update(frame):
        ax.clear()
        current_state = consensus_over_time[frame]
        heights = current_state[:, 0].tolist()
        tips = current_state[:, 1].tolist()

        colors = [tip_to_color[tip] for tip in tips]
        ax.bar(range(num_nodes), heights, color=colors)
        ax.set_ylim(0, max([s[:,0].max().item() for s in consensus_over_time]) + 1)
        ax.set_title(f"Timestep {frame}")
        ax.set_xlabel("Node")
        ax.set_ylabel("Chain Height")

    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=interval, repeat=False)
    plt.close(fig)
    return HTML(ani.to_jshtml())


def generate_malicious_mask(num_nodes, fraction=0.2):
    malicious_mask = torch.zeros(num_nodes, dtype=torch.bool)
    indices = torch.randperm(num_nodes)[:int(num_nodes * fraction)]
    malicious_mask[indices] = True
    return malicious_mask