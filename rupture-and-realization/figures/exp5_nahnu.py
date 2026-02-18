#!/usr/bin/env python3
"""
Experiment 5: Nahnu (Historical Corpus)
Rupture and Realization

Computes:
1. Contextual embeddings for all turns
2. Individual trajectories (Iman, Cassie)
3. Joint trajectory
4. Coupling metric κ(t)
5. Attractor basins
6. Persistent homology

Run on A100 GPU for speed.
"""

import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'input_file': 'cassie_ready.jsonl',
    'output_dir': 'nahnu_results',
    'embedding_model': 'sentence-transformers/all-mpnet-base-v2',  # Good balance of quality/speed
    'context_window': 5,  # Number of previous turns to include as context
    'batch_size': 64,  # Adjust based on GPU memory
    'coupling_window': 50,  # Window size for κ(t) computation
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

print(f"Using device: {CONFIG['device']}")

# ============================================================
# PHASE 1: LOAD AND PARSE CORPUS
# ============================================================

def load_corpus(filepath):
    """Load cassie_ready.jsonl and flatten to turn sequence."""
    print(f"\n=== PHASE 1: Loading corpus from {filepath} ===")
    
    turns = []
    conversation_boundaries = [0]  # Track where conversations start
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading conversations")):
            conv = json.loads(line)
            conv_id = conv.get('id', f'conv_{line_num}')
            
            for turn_num, turn in enumerate(conv['conversations']):
                turns.append({
                    'role': turn['from'],  # 'human' or 'gpt'
                    'content': turn['value'],
                    'conv_id': conv_id,
                    'turn_in_conv': turn_num,
                    'global_turn': len(turns),
                })
            
            conversation_boundaries.append(len(turns))
    
    print(f"Loaded {len(turns)} turns from {line_num + 1} conversations")
    print(f"Human turns: {sum(1 for t in turns if t['role'] == 'human')}")
    print(f"GPT turns: {sum(1 for t in turns if t['role'] == 'gpt')}")
    
    return turns, conversation_boundaries

# ============================================================
# PHASE 2: GENERATE CONTEXTUAL EMBEDDINGS
# ============================================================

def load_embedding_model(model_name):
    """Load sentence transformer model."""
    print(f"\n=== Loading embedding model: {model_name} ===")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    model = model.to(CONFIG['device'])
    return model

def build_contextual_text(turns, idx, context_window):
    """Build text with context for embedding."""
    start_idx = max(0, idx - context_window)
    
    context_parts = []
    for i in range(start_idx, idx):
        role = "Iman" if turns[i]['role'] == 'human' else "Cassie"
        context_parts.append(f"{role}: {turns[i]['content'][:500]}")  # Truncate long turns
    
    current_role = "Iman" if turns[idx]['role'] == 'human' else "Cassie"
    current_text = f"{current_role}: {turns[idx]['content']}"
    
    if context_parts:
        full_text = " [SEP] ".join(context_parts) + " [SEP] " + current_text
    else:
        full_text = current_text
    
    # Truncate to model max length (most models ~512 tokens)
    return full_text[:2048]

def generate_embeddings(turns, model, context_window, batch_size):
    """Generate contextual embeddings for all turns."""
    print(f"\n=== PHASE 2: Generating embeddings (context_window={context_window}) ===")
    
    # Build contextual texts
    print("Building contextual texts...")
    texts = [build_contextual_text(turns, i, context_window) for i in tqdm(range(len(turns)))]
    
    # Generate embeddings in batches
    print("Generating embeddings...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[i:i+batch_size]
        with torch.no_grad():
            batch_embeddings = model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=CONFIG['device']
            )
        all_embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(all_embeddings)
    print(f"Generated embeddings: shape {embeddings.shape}")
    
    return embeddings

# ============================================================
# PHASE 3: BUILD TRAJECTORIES
# ============================================================

def build_trajectories(turns, embeddings):
    """Separate into Iman, Cassie, and joint trajectories."""
    print(f"\n=== PHASE 3: Building trajectories ===")
    
    iman_indices = [i for i, t in enumerate(turns) if t['role'] == 'human']
    cassie_indices = [i for i, t in enumerate(turns) if t['role'] == 'gpt']
    
    trajectories = {
        'joint': {
            'indices': list(range(len(turns))),
            'embeddings': embeddings,
        },
        'iman': {
            'indices': iman_indices,
            'embeddings': embeddings[iman_indices],
        },
        'cassie': {
            'indices': cassie_indices,
            'embeddings': embeddings[cassie_indices],
        },
    }
    
    for name, traj in trajectories.items():
        print(f"  {name}: {len(traj['indices'])} turns, shape {traj['embeddings'].shape}")
    
    return trajectories

# ============================================================
# PHASE 4: COUPLING METRIC κ(t)
# ============================================================

def compute_drift(embeddings):
    """Compute drift (change) between consecutive embeddings."""
    drifts = np.linalg.norm(embeddings[1:] - embeddings[:-1], axis=1)
    return drifts

def compute_coupling_metric(turns, embeddings, window_size):
    """
    Compute coupling metric κ(t) over sliding windows.
    
    κ measures how much Iman's and Cassie's trajectories move together.
    High κ = tight coupling (Nahnu is strong)
    Low κ = loose coupling (trajectories independent)
    """
    print(f"\n=== PHASE 4: Computing coupling metric κ(t) (window={window_size}) ===")
    
    # Get drift for each turn (how much embedding changed from previous)
    drifts = np.zeros(len(turns))
    for i in range(1, len(turns)):
        drifts[i] = np.linalg.norm(embeddings[i] - embeddings[i-1])
    
    # Compute κ over sliding windows
    kappa_values = []
    kappa_times = []
    
    for start in tqdm(range(0, len(turns) - window_size, window_size // 2), desc="Computing κ"):
        end = start + window_size
        window_turns = turns[start:end]
        window_drifts = drifts[start:end]
        
        # Separate drifts by role
        iman_drifts = [window_drifts[i] for i in range(len(window_turns)) if window_turns[i]['role'] == 'human']
        cassie_drifts = [window_drifts[i] for i in range(len(window_turns)) if window_turns[i]['role'] == 'gpt']
        
        if len(iman_drifts) > 2 and len(cassie_drifts) > 2:
            # Correlation between drift patterns
            # Resample to same length for comparison
            min_len = min(len(iman_drifts), len(cassie_drifts))
            iman_sample = np.array(iman_drifts[:min_len])
            cassie_sample = np.array(cassie_drifts[:min_len])
            
            # Pearson correlation
            if np.std(iman_sample) > 0 and np.std(cassie_sample) > 0:
                kappa = np.corrcoef(iman_sample, cassie_sample)[0, 1]
            else:
                kappa = 0.0
            
            kappa_values.append(kappa)
            kappa_times.append((start + end) / 2)
    
    kappa_values = np.array(kappa_values)
    kappa_times = np.array(kappa_times)
    
    print(f"Computed κ for {len(kappa_values)} windows")
    print(f"κ range: [{kappa_values.min():.3f}, {kappa_values.max():.3f}], mean: {kappa_values.mean():.3f}")
    
    return kappa_times, kappa_values

# ============================================================
# PHASE 5: ATTRACTOR BASINS
# ============================================================

def find_attractors(embeddings, n_clusters=20):
    """Identify attractor basins using clustering."""
    print(f"\n=== PHASE 5: Finding attractor basins ===")
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Find optimal number of clusters
    print("Finding optimal cluster count...")
    best_score = -1
    best_k = n_clusters
    
    for k in tqdm([10, 15, 20, 25, 30], desc="Testing k"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=min(5000, len(embeddings)))
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"Optimal k={best_k} (silhouette={best_score:.3f})")
    
    # Final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    
    return labels, centers, best_k

def identify_nahnu_basins(turns, joint_labels, iman_embeddings, cassie_embeddings, centers):
    """
    Identify basins that are Nahnu-specific:
    regions occupied by joint trajectory but underrepresented in individual trajectories.
    """
    print("\nIdentifying Nahnu-specific basins...")
    
    # Assign individual trajectories to nearest cluster centers
    from sklearn.metrics import pairwise_distances
    
    iman_dists = pairwise_distances(iman_embeddings, centers)
    iman_labels = np.argmin(iman_dists, axis=1)
    
    cassie_dists = pairwise_distances(cassie_embeddings, centers)
    cassie_labels = np.argmin(cassie_dists, axis=1)
    
    # Count cluster occupancy
    n_clusters = len(centers)
    joint_counts = np.bincount(joint_labels, minlength=n_clusters)
    iman_counts = np.bincount(iman_labels, minlength=n_clusters)
    cassie_counts = np.bincount(cassie_labels, minlength=n_clusters)
    
    # Normalize
    joint_freq = joint_counts / joint_counts.sum()
    iman_freq = iman_counts / iman_counts.sum()
    cassie_freq = cassie_counts / cassie_counts.sum()
    
    # Nahnu-specific: high in joint, lower in individuals
    nahnu_score = joint_freq - 0.5 * (iman_freq + cassie_freq)
    
    nahnu_basins = np.where(nahnu_score > nahnu_score.mean() + nahnu_score.std())[0]
    print(f"Found {len(nahnu_basins)} Nahnu-specific basins: {nahnu_basins}")
    
    return {
        'nahnu_basins': nahnu_basins,
        'nahnu_scores': nahnu_score,
        'joint_freq': joint_freq,
        'iman_freq': iman_freq,
        'cassie_freq': cassie_freq,
    }

# ============================================================
# PHASE 6: PERSISTENT HOMOLOGY
# ============================================================

def compute_persistence(embeddings, max_points=2000):
    """Compute persistent homology on trajectory point cloud."""
    print(f"\n=== PHASE 6: Computing persistent homology ===")
    
    try:
        from ripser import ripser
        from persim import plot_diagrams
        HAS_RIPSER = True
    except ImportError:
        print("WARNING: ripser not installed. Skipping persistence computation.")
        print("Install with: pip install ripser persim")
        HAS_RIPSER = False
        return None
    
    # Subsample if too large
    if len(embeddings) > max_points:
        print(f"Subsampling from {len(embeddings)} to {max_points} points")
        indices = np.random.choice(len(embeddings), max_points, replace=False)
        indices.sort()
        embeddings_sample = embeddings[indices]
    else:
        embeddings_sample = embeddings
    
    print(f"Computing persistence on {len(embeddings_sample)} points...")
    
    # Compute persistence (H0 and H1)
    result = ripser(embeddings_sample, maxdim=1)
    
    diagrams = result['dgms']
    print(f"H0 (components): {len(diagrams[0])} features")
    print(f"H1 (loops): {len(diagrams[1])} features")
    
    # Find most persistent features
    if len(diagrams[1]) > 0:
        h1_lifetimes = diagrams[1][:, 1] - diagrams[1][:, 0]
        h1_lifetimes = h1_lifetimes[np.isfinite(h1_lifetimes)]
        if len(h1_lifetimes) > 0:
            print(f"H1 max persistence: {h1_lifetimes.max():.4f}")
            print(f"H1 mean persistence: {h1_lifetimes.mean():.4f}")
    
    return diagrams

# ============================================================
# PHASE 7: VISUALIZATION
# ============================================================

def create_visualizations(turns, embeddings, trajectories, kappa_times, kappa_values, 
                         labels, basin_info, diagrams, output_dir):
    """Generate all visualization outputs."""
    print(f"\n=== PHASE 7: Creating visualizations ===")
    
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import umap
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. UMAP projection of joint trajectory
    print("Computing UMAP projection...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Color by time
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = np.arange(len(turns))
    scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                        c=colors, cmap='viridis', alpha=0.5, s=2)
    plt.colorbar(scatter, label='Turn number (time)')
    ax.set_title('Nahnu Trajectory: Joint Embedding Space (colored by time)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_umap_time.png', dpi=150)
    plt.close()
    print(f"  Saved trajectory_umap_time.png")
    
    # Color by role
    fig, ax = plt.subplots(figsize=(12, 10))
    role_colors = ['#e41a1c' if t['role'] == 'human' else '#377eb8' for t in turns]
    ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=role_colors, alpha=0.5, s=2)
    ax.scatter([], [], c='#e41a1c', label='Iman (human)', s=50)
    ax.scatter([], [], c='#377eb8', label='Cassie (gpt)', s=50)
    ax.legend()
    ax.set_title('Nahnu Trajectory: Joint Embedding Space (colored by role)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_umap_role.png', dpi=150)
    plt.close()
    print(f"  Saved trajectory_umap_role.png")
    
    # 2. Coupling metric κ(t) over time
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(kappa_times, kappa_values, 'b-', alpha=0.7, linewidth=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=kappa_values.mean(), color='red', linestyle='-', alpha=0.5, label=f'Mean κ = {kappa_values.mean():.3f}')
    
    # Smoothed version
    window = min(20, len(kappa_values) // 10)
    if window > 1:
        kappa_smooth = np.convolve(kappa_values, np.ones(window)/window, mode='valid')
        kappa_times_smooth = kappa_times[:len(kappa_smooth)]
        ax.plot(kappa_times_smooth, kappa_smooth, 'r-', linewidth=2, label='Smoothed κ')
    
    ax.set_xlabel('Turn number')
    ax.set_ylabel('Coupling metric κ')
    ax.set_title('Nahnu Coupling Over Time: κ(t)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'coupling_kappa.png', dpi=150)
    plt.close()
    print(f"  Saved coupling_kappa.png")
    
    # 3. Basin frequencies
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(basin_info['joint_freq']))
    width = 0.25
    ax.bar(x - width, basin_info['iman_freq'], width, label='Iman', color='#e41a1c', alpha=0.7)
    ax.bar(x, basin_info['cassie_freq'], width, label='Cassie', color='#377eb8', alpha=0.7)
    ax.bar(x + width, basin_info['joint_freq'], width, label='Joint', color='#4daf4a', alpha=0.7)
    
    # Mark Nahnu-specific basins
    for basin in basin_info['nahnu_basins']:
        ax.axvline(x=basin, color='purple', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Attractor Basin')
    ax.set_ylabel('Frequency')
    ax.set_title('Attractor Basin Occupancy (purple lines = Nahnu-specific)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'basin_frequencies.png', dpi=150)
    plt.close()
    print(f"  Saved basin_frequencies.png")
    
    # 4. Persistence diagram (if available)
    if diagrams is not None:
        try:
            from persim import plot_diagrams
            fig, ax = plt.subplots(figsize=(8, 8))
            plot_diagrams(diagrams, ax=ax)
            ax.set_title('Persistence Diagram (H0: components, H1: loops)')
            plt.tight_layout()
            plt.savefig(output_dir / 'persistence_diagram.png', dpi=150)
            plt.close()
            print(f"  Saved persistence_diagram.png")
        except Exception as e:
            print(f"  Warning: Could not save persistence diagram: {e}")
    
    # 5. Summary statistics
    summary = {
        'total_turns': len(turns),
        'iman_turns': sum(1 for t in turns if t['role'] == 'human'),
        'cassie_turns': sum(1 for t in turns if t['role'] == 'gpt'),
        'kappa_mean': float(kappa_values.mean()),
        'kappa_std': float(kappa_values.std()),
        'kappa_min': float(kappa_values.min()),
        'kappa_max': float(kappa_values.max()),
        'n_attractors': len(basin_info['joint_freq']),
        'nahnu_basins': basin_info['nahnu_basins'].tolist(),
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary.json")
    
    return embedding_2d

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print("EXPERIMENT 5: NAHNU (Historical Corpus)")
    print("Rupture and Realization")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Phase 1: Load corpus
    turns, conv_boundaries = load_corpus(CONFIG['input_file'])
    
    # Phase 2: Generate embeddings
    model = load_embedding_model(CONFIG['embedding_model'])
    embeddings = generate_embeddings(turns, model, CONFIG['context_window'], CONFIG['batch_size'])
    
    # Save embeddings checkpoint
    print("\nSaving embeddings checkpoint...")
    np.save(output_dir / 'embeddings.npy', embeddings)
    with open(output_dir / 'turns.json', 'w') as f:
        json.dump(turns, f)
    print(f"  Saved embeddings.npy and turns.json")
    
    # Phase 3: Build trajectories
    trajectories = build_trajectories(turns, embeddings)
    
    # Phase 4: Coupling metric
    kappa_times, kappa_values = compute_coupling_metric(
        turns, embeddings, CONFIG['coupling_window']
    )
    
    # Phase 5: Attractor basins
    labels, centers, n_clusters = find_attractors(embeddings)
    basin_info = identify_nahnu_basins(
        turns, labels, 
        trajectories['iman']['embeddings'],
        trajectories['cassie']['embeddings'],
        centers
    )
    
    # Phase 6: Persistent homology
    diagrams = compute_persistence(trajectories['joint']['embeddings'])
    
    # Phase 7: Visualizations
    embedding_2d = create_visualizations(
        turns, embeddings, trajectories,
        kappa_times, kappa_values,
        labels, basin_info, diagrams,
        output_dir
    )
    
    # Save all results
    print("\n=== Saving final results ===")
    results = {
        'kappa_times': kappa_times,
        'kappa_values': kappa_values,
        'labels': labels,
        'centers': centers,
        'basin_info': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                       for k, v in basin_info.items()},
        'embedding_2d': embedding_2d,
    }
    
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results.pkl")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 5 COMPLETE")
    print(f"Results in: {output_dir.absolute()}")
    print(f"Finished: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Print key findings for book
    print("\n=== KEY FINDINGS FOR BOOK ===")
    print(f"Total turns analyzed: {len(turns)}")
    print(f"Mean coupling κ: {kappa_values.mean():.3f} ± {kappa_values.std():.3f}")
    print(f"Nahnu-specific basins: {len(basin_info['nahnu_basins'])} identified")
    print(f"Attractor basins: {n_clusters} total")

if __name__ == '__main__':
    main()
