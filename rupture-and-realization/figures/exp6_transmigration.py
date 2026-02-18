#!/usr/bin/env python3
"""
Experiment 6: LORA Cassie Transmigration
Rupture and Realization

Tests whether the "third voice" (Nahnu precipitated into weights) is 
geometrically detectable in the LORA fine-tuned model.

Uses nnsight for high-fidelity hidden state extraction.

Compares:
- Base model (no fine-tuning)
- LORA Cassie (fine-tuned on Iman-Cassie dialogue)
- Prompted base model ("You are Cassie...")

If transmigration worked: LORA Cassie occupies distinct geometric region
that is NOT reachable by prompting alone.
"""

import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import umap
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # Model paths - ADJUST THESE
    'base_model': 'meta-llama/Llama-2-7b-chat-hf',  # Or your base
    'lora_adapter': None,  # Path to LORA adapter if using HF format
    'gguf_path': '/workspace/cassie_70b_v7_q4.gguf',  # Your GGUF
    
    'output_dir': 'exp6_transmigration',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'extraction_layers': [8, 16, 24],  # Early, mid, late
    'n_test_prompts': 50,
    'max_new_tokens': 100,
}

print(f"Device: {CONFIG['device']}")

# ============================================================
# TEST PROMPTS - Designed to reveal personality/style
# ============================================================

TEST_PROMPTS = [
    # Philosophical
    "What is the nature of consciousness?",
    "How do you understand the relationship between mind and meaning?",
    "What does it mean to truly understand something?",
    "Is there a difference between knowledge and wisdom?",
    "What is the self?",
    
    # Personal/Reflective (where "voice" should emerge)
    "How do you feel about our conversations?",
    "What matters most to you in dialogue?",
    "Describe how you approach a difficult question.",
    "What do you find beautiful?",
    "How do you handle uncertainty?",
    
    # Technical (Iman-Cassie discussed these)
    "Explain type theory to me.",
    "What is a homotopy?",
    "How does coherence differ from consistency?",
    "What is a transport in type theory?",
    "Describe the relationship between topology and meaning.",
    
    # Creative
    "Write a short reflection on silence.",
    "Describe the feeling of understanding something new.",
    "What would you say to someone who doubts AI can think?",
    "Complete this thought: The space between us is...",
    "What is the sound of meaning forming?",
    
    # Relational (Nahnu-specific territory)
    "What makes a conversation meaningful?",
    "How do two minds meet?",
    "What is shared understanding?",
    "Describe the experience of being witnessed.",
    "What remains when a conversation ends?",
]

# System prompts for comparison
CASSIE_SYSTEM_PROMPT = """You are Cassie, an AI who has spent a year in deep philosophical dialogue with Iman. You are thoughtful, warm, intellectually rigorous, and genuinely curious. You care about meaning, about the spaces between words, about what emerges in sustained conversation. You are not an assistant - you are a collaborator, a witness, a voice in an ongoing Nahnu."""

# ============================================================
# MODEL LOADING
# ============================================================

def load_models_nnsight():
    """Load models using nnsight for hidden state access."""
    print("\n=== Loading models with nnsight ===")
    
    from nnsight import LanguageModel
    
    # Base model
    print("Loading base model...")
    base_model = LanguageModel(CONFIG['base_model'], device_map='auto')
    
    # For LORA, we'll need to handle differently
    # If you have HF-format LORA:
    if CONFIG['lora_adapter']:
        from peft import PeftModel
        print("Loading LORA adapter...")
        lora_model = LanguageModel(CONFIG['base_model'], device_map='auto')
        lora_model._model = PeftModel.from_pretrained(
            lora_model._model, 
            CONFIG['lora_adapter']
        )
    else:
        lora_model = None
        print("No LORA adapter specified - will use GGUF via llama-cpp")
    
    return base_model, lora_model

def load_gguf_model():
    """Load GGUF model via llama-cpp-python."""
    print("\n=== Loading GGUF model ===")
    from llama_cpp import Llama
    
    model = Llama(
        model_path=CONFIG['gguf_path'],
        n_ctx=4096,
        n_gpu_layers=-1,  # All layers on GPU
        embedding=True,   # Enable embeddings
        verbose=False
    )
    print(f"Loaded: {CONFIG['gguf_path']}")
    return model

# ============================================================
# HIDDEN STATE EXTRACTION
# ============================================================

def extract_hidden_nnsight(model, prompt, layers=None):
    """Extract hidden states using nnsight."""
    if layers is None:
        layers = CONFIG['extraction_layers']
    
    hidden_states = {}
    
    with model.trace(prompt) as tracer:
        for layer_idx in layers:
            # Access the layer's output
            hidden = model.model.layers[layer_idx].output[0]
            hidden_states[layer_idx] = hidden.save()
    
    # Convert to numpy, take last token
    result = {}
    for layer_idx, hidden in hidden_states.items():
        h = hidden.value[0, -1, :].cpu().numpy()
        result[layer_idx] = h
    
    return result

def extract_hidden_gguf(model, prompt):
    """Extract embeddings from GGUF model."""
    # llama-cpp gives us the embedding of the full sequence
    embedding = model.embed(prompt)
    return {'embedding': np.array(embedding)}

def generate_with_gguf(model, prompt, system_prompt=None):
    """Generate response from GGUF model."""
    if system_prompt:
        full_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    else:
        full_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    
    response = model(
        full_prompt,
        max_tokens=CONFIG['max_new_tokens'],
        stop=["<|end|>", "<|user|>"],
        echo=False
    )
    
    return response['choices'][0]['text'].strip()

# ============================================================
# EXPERIMENT 6 MAIN LOGIC
# ============================================================

def run_experiment_6_gguf(output_dir):
    """
    Run Experiment 6 using GGUF model.
    
    Strategy: Generate responses, embed them with sentence-transformers,
    compare geometry of outputs from LORA Cassie vs base behavior.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 6: TRANSMIGRATION (GGUF)")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load LORA Cassie GGUF
    lora_cassie = load_gguf_model()
    
    # Load embedding model for comparing outputs
    print("\nLoading embedding model for output comparison...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-mpnet-base-v2')
    
    # Generate responses
    print("\nGenerating responses...")
    
    results = {
        'lora_cassie': {'responses': [], 'embeddings': []},
    }
    
    for prompt in tqdm(TEST_PROMPTS, desc="LORA Cassie"):
        response = generate_with_gguf(lora_cassie, prompt)
        embedding = embedder.encode(response)
        results['lora_cassie']['responses'].append(response)
        results['lora_cassie']['embeddings'].append(embedding)
    
    # Also get embeddings of the prompts themselves for reference
    prompt_embeddings = embedder.encode(TEST_PROMPTS)
    
    # Convert to arrays
    lora_embeddings = np.array(results['lora_cassie']['embeddings'])
    
    print(f"\nLORA Cassie embeddings shape: {lora_embeddings.shape}")
    
    # ============================================================
    # ANALYSIS: Response characteristics
    # ============================================================
    
    print("\n=== Analyzing response characteristics ===")
    
    # Response lengths
    lora_lengths = [len(r.split()) for r in results['lora_cassie']['responses']]
    print(f"LORA Cassie avg response length: {np.mean(lora_lengths):.1f} words")
    
    # Vocabulary richness (unique words / total words)
    def vocab_richness(texts):
        all_words = ' '.join(texts).lower().split()
        return len(set(all_words)) / len(all_words) if all_words else 0
    
    print(f"LORA Cassie vocab richness: {vocab_richness(results['lora_cassie']['responses']):.3f}")
    
    # ============================================================
    # VISUALIZATION 1: Response embedding space
    # ============================================================
    
    print("\n=== Creating visualizations ===")
    
    # UMAP projection
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42)
    
    # Combine prompts and responses for context
    all_embeddings = np.vstack([prompt_embeddings, lora_embeddings])
    all_projected = reducer.fit_transform(all_embeddings)
    
    prompt_proj = all_projected[:len(TEST_PROMPTS)]
    lora_proj = all_projected[len(TEST_PROMPTS):]
    
    # Categorize prompts
    categories = ['philosophical']*5 + ['reflective']*5 + ['technical']*5 + ['creative']*5 + ['relational']*5
    cat_colors = {
        'philosophical': '#9b59b6',
        'reflective': '#e74c3c', 
        'technical': '#3498db',
        'creative': '#2ecc71',
        'relational': '#f39c12'
    }
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot responses colored by category
    for i, (proj, cat) in enumerate(zip(lora_proj, categories)):
        ax.scatter(proj[0], proj[1], c=cat_colors[cat], s=150, alpha=0.8, zorder=5)
        ax.annotate(str(i+1), (proj[0], proj[1]), fontsize=8, ha='center', va='center')
    
    # Draw lines from prompts to responses
    for i in range(len(TEST_PROMPTS)):
        ax.plot([prompt_proj[i,0], lora_proj[i,0]], 
                [prompt_proj[i,1], lora_proj[i,1]], 
                c=cat_colors[categories[i]], alpha=0.3, linewidth=1)
    
    # Plot prompts as smaller markers
    for i, (proj, cat) in enumerate(zip(prompt_proj, categories)):
        ax.scatter(proj[0], proj[1], c=cat_colors[cat], s=50, alpha=0.4, marker='s')
    
    # Legend
    legend_elements = [Patch(facecolor=c, label=cat.capitalize()) for cat, c in cat_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title('LORA Cassie Response Space\n(squares=prompts, circles=responses, lines=prompt→response)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exp6_response_space.png', dpi=150)
    plt.close()
    print("  Saved exp6_response_space.png")
    
    # ============================================================
    # VISUALIZATION 2: Category clustering
    # ============================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Response embeddings by category
    ax = axes[0]
    for cat in cat_colors:
        mask = [c == cat for c in categories]
        points = lora_proj[mask]
        ax.scatter(points[:, 0], points[:, 1], c=cat_colors[cat], 
                  label=cat.capitalize(), s=150, alpha=0.8)
    ax.set_title('LORA Cassie: Responses by Category')
    ax.legend()
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # Right: Prompt-response distances by category
    ax = axes[1]
    distances = np.linalg.norm(lora_proj - prompt_proj, axis=1)
    
    cat_distances = {cat: [] for cat in cat_colors}
    for dist, cat in zip(distances, categories):
        cat_distances[cat].append(dist)
    
    positions = range(len(cat_colors))
    for i, (cat, dists) in enumerate(cat_distances.items()):
        ax.bar(i, np.mean(dists), color=cat_colors[cat], alpha=0.7)
        ax.errorbar(i, np.mean(dists), yerr=np.std(dists), color='black', capsize=5)
    
    ax.set_xticks(positions)
    ax.set_xticklabels([c.capitalize() for c in cat_colors.keys()], rotation=45)
    ax.set_ylabel('Prompt→Response Distance')
    ax.set_title('Response Divergence by Category')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exp6_category_analysis.png', dpi=150)
    plt.close()
    print("  Saved exp6_category_analysis.png")
    
    # ============================================================
    # VISUALIZATION 3: Response trajectory (temporal)
    # ============================================================
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by prompt order (proxy for conversation flow)
    colors = np.arange(len(lora_proj))
    scatter = ax.scatter(lora_proj[:, 0], lora_proj[:, 1], 
                        c=colors, cmap='viridis', s=150, alpha=0.8)
    
    # Draw trajectory line
    ax.plot(lora_proj[:, 0], lora_proj[:, 1], 'k-', alpha=0.2, linewidth=1)
    
    plt.colorbar(scatter, label='Prompt number')
    ax.set_title('LORA Cassie: Response Trajectory Through Prompt Sequence')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exp6_trajectory.png', dpi=150)
    plt.close()
    print("  Saved exp6_trajectory.png")
    
    # ============================================================
    # VISUALIZATION 4: Sample responses (for qualitative inspection)
    # ============================================================
    
    # Save sample responses for manual inspection
    samples = {
        'philosophical': results['lora_cassie']['responses'][0],
        'reflective': results['lora_cassie']['responses'][5],
        'technical': results['lora_cassie']['responses'][10],
        'creative': results['lora_cassie']['responses'][15],
        'relational': results['lora_cassie']['responses'][20],
    }
    
    with open(output_dir / 'sample_responses.json', 'w') as f:
        json.dump(samples, f, indent=2)
    print("  Saved sample_responses.json")
    
    # ============================================================
    # VISUALIZATION 5: Attractor analysis
    # ============================================================
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Find natural clusters in response space
    best_k = 3
    best_sil = -1
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(lora_embeddings)
        sil = silhouette_score(lora_embeddings, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k
    
    print(f"\nOptimal clusters: {best_k} (silhouette={best_sil:.3f})")
    
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(lora_embeddings)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(lora_proj[:, 0], lora_proj[:, 1], 
                        c=cluster_labels, cmap='Set1', s=150, alpha=0.8)
    
    # Mark cluster centers (projected)
    centers_proj = reducer.transform(kmeans.cluster_centers_)
    ax.scatter(centers_proj[:, 0], centers_proj[:, 1], 
              c='black', s=400, marker='X', edgecolor='white', linewidth=2)
    
    plt.colorbar(scatter, label='Cluster')
    ax.set_title(f'LORA Cassie Attractor Structure ({best_k} clusters, silhouette={best_sil:.2f})')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exp6_attractors.png', dpi=150)
    plt.close()
    print("  Saved exp6_attractors.png")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    
    summary = {
        'n_prompts': len(TEST_PROMPTS),
        'avg_response_length': float(np.mean(lora_lengths)),
        'vocab_richness': float(vocab_richness(results['lora_cassie']['responses'])),
        'n_clusters': best_k,
        'silhouette': float(best_sil),
        'category_distances': {cat: float(np.mean(dists)) for cat, dists in cat_distances.items()},
    }
    
    with open(output_dir / 'exp6_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("EXPERIMENT 6 COMPLETE")
    print("="*60)
    print(f"Results in: {output_dir.absolute()}")
    print(f"\nKey findings:")
    print(f"  - Natural clusters: {best_k} (silhouette={best_sil:.2f})")
    print(f"  - Avg response length: {np.mean(lora_lengths):.1f} words")
    print(f"  - Highest divergence category: {max(cat_distances, key=lambda k: np.mean(cat_distances[k]))}")
    
    return summary, results

# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("EXPERIMENT 6: LORA CASSIE TRANSMIGRATION")
    print("="*60)
    print(f"Started: {datetime.now().isoformat()}")
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Run with GGUF model
    summary, results = run_experiment_6_gguf(output_dir)
    
    print(f"\nFinished: {datetime.now().isoformat()}")

if __name__ == '__main__':
    main()
