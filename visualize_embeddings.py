import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from scipy.linalg import eigh
import umap
import os
from utils import read_txt
# Suppress warnings from TensorFlow and related libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

def construct_ligand_profile(edge_file, num_targets, num_ingredients, use_tfidf=True):
    """
    Construct ligand profile matrix

    Args:
        edge_file: Path to edge file
        num_targets: Number of targets
        num_ingredients: Number of ingredients
        use_tfidf: Whether to use TF-IDF weighting

    Returns:
        profile_matrix: Target x Ingredient ligand profile matrix
    """
    # Read edges
    edges = read_txt(edge_file)

    # Construct binary matrix
    M = np.zeros((num_targets, num_ingredients))
    for src, dst in edges:
        M[dst, src] = 1  # dst is target, src is ingredient

    if use_tfidf:
        # TF: Row normalization
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        TF = M / row_sums

        # IDF: log(N / df)
        N = num_targets
        df = M.sum(axis=0) + 1  # +1 to avoid division by zero
        IDF = np.log(N / df)

        # TF-IDF
        profile_matrix = TF * IDF
    else:
        profile_matrix = M

    return profile_matrix


def detect_isolated_nodes(embeddings):
    """
    Detect isolated nodes (zero vectors)

    Args:
        embeddings: Node embedding matrix

    Returns:
        non_isolated_mask: Boolean mask for non-isolated nodes
        isolated_indices: Indices of isolated nodes
    """
    # Detect zero vectors
    zero_mask = np.all(embeddings == 0, axis=1)
    non_isolated_mask = ~zero_mask
    isolated_indices = np.where(zero_mask)[0]

    print(f"  Detected {zero_mask.sum()} isolated nodes (zero vectors)")
    print(f"  Keeping {non_isolated_mask.sum()} non-isolated nodes for analysis")

    return non_isolated_mask, isolated_indices


def compute_similarity_matrices(profile_matrix, target_embeddings, exclude_isolated=True):
    """
    Compute similarity matrices: Ligand profile and Embeddings

    Args:
        profile_matrix: Ligand profile matrix
        target_embeddings: Target embeddings
        exclude_isolated: Whether to exclude isolated nodes

    Returns:
        ligand_sim: Ligand profile similarity matrix
        embedding_sim: Embedding similarity matrix
        non_isolated_mask: Non-isolated node mask (if exclude_isolated=True)
    """
    if exclude_isolated:
        # Detect and exclude isolated nodes
        non_isolated_mask, isolated_indices = detect_isolated_nodes(target_embeddings)

        if non_isolated_mask.sum() == 0:
            print("Warning: All nodes are isolated!")
            return None, None, None

        # Filter ligand profile and embeddings
        profile_matrix_filtered = profile_matrix[non_isolated_mask, :]
        target_embeddings_filtered = target_embeddings[non_isolated_mask, :]

        # Compute similarity
        profile_norm = normalize(profile_matrix_filtered, axis=1, norm='l2')
        ligand_sim = cosine_similarity(profile_norm)

        embedding_norm = normalize(target_embeddings_filtered, axis=1, norm='l2')
        embedding_sim = cosine_similarity(embedding_norm)

        return ligand_sim, embedding_sim, non_isolated_mask
    else:
        # Do not exclude isolated nodes (original method)
        profile_norm = normalize(profile_matrix, axis=1, norm='l2')
        ligand_sim = cosine_similarity(profile_norm)

        embedding_norm = normalize(target_embeddings, axis=1, norm='l2')
        embedding_sim = cosine_similarity(embedding_norm)

        return ligand_sim, embedding_sim, None


def calculate_similarity_correlation(ligand_sim, embedding_sim, exclude_diagonal=True):
    """
    Calculate Spearman correlation between two similarity matrices

    Args:
        ligand_sim: Ligand profile similarity matrix
        embedding_sim: Embedding similarity matrix
        exclude_diagonal: Whether to exclude diagonal elements

    Returns:
        correlation: Spearman correlation coefficient
        p_value: p-value
        n_pairs: Number of data points used for calculation
    """
    # Ensure matrices have same shape
    assert ligand_sim.shape == embedding_sim.shape, "Matrix shapes must match"
    assert ligand_sim.shape[0] == ligand_sim.shape[1], "Must be square matrices"

    n = ligand_sim.shape[0]

    if exclude_diagonal:
        # Get upper triangle indices (excluding diagonal)
        # k=1 means start from the first diagonal above the main diagonal
        upper_tri_indices = np.triu_indices(n, k=1)

        # Extract upper triangle values
        ligand_values = ligand_sim[upper_tri_indices]
        embedding_values = embedding_sim[upper_tri_indices]
    else:
        # Use all values (including diagonal)
        ligand_values = ligand_sim.flatten()
        embedding_values = embedding_sim.flatten()

    # Compute Spearman correlation
    if len(ligand_values) > 0:
        correlation, p_value = spearmanr(ligand_values, embedding_values)
        n_pairs = len(ligand_values)
    else:
        correlation = np.nan
        p_value = np.nan
        n_pairs = 0

    return correlation, p_value, n_pairs


def analyze_similarity_at_thresholds(ligand_sim, embedding_sim,
                                     thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    Analyze correlation at different similarity thresholds
    """
    results = {}
    n = ligand_sim.shape[0]
    total_pairs = n * (n - 1) // 2  # Number of upper triangle elements

    for threshold in thresholds:
        # Get upper triangle indices
        upper_tri_indices = np.triu_indices(n, k=1)
        ligand_values = ligand_sim[upper_tri_indices]
        embedding_values = embedding_sim[upper_tri_indices]

        # Filter ligand profile similarities above threshold
        high_sim_mask = ligand_values >= threshold

        if high_sim_mask.sum() > 0:
            filtered_ligand = ligand_values[high_sim_mask]
            filtered_embedding = embedding_values[high_sim_mask]

            correlation, p_value = spearmanr(filtered_ligand, filtered_embedding)

            results[threshold] = {
                'correlation': correlation,
                'p_value': p_value,
                'n_pairs': high_sim_mask.sum(),
                'percentage': high_sim_mask.sum() / total_pairs * 100
            }
        else:
            results[threshold] = {
                'correlation': np.nan,
                'p_value': np.nan,
                'n_pairs': 0,
                'percentage': 0
            }

    return results


def create_similarity_heatmaps(ligand_sim, embedding_sim_before, embedding_sim_after,
                               output_dir='UMAP'):
    """
    Create comparison heatmaps of similarity matrices
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ligand Profile Similarity
    sns.heatmap(ligand_sim, cmap='coolwarm', center=0,
                vmin=-1, vmax=1, ax=axes[0], cbar_kws={'label': 'Similarity'})
    axes[0].set_title('Ligand Profile Similarity')
    axes[0].set_xlabel('Target Index')
    axes[0].set_ylabel('Target Index')

    # Embedding Similarity Before Training
    sns.heatmap(embedding_sim_before, cmap='coolwarm', center=0,
                vmin=-1, vmax=1, ax=axes[1], cbar_kws={'label': 'Similarity'})
    axes[1].set_title('Embedding Similarity (Before Training)')
    axes[1].set_xlabel('Target Index')
    axes[1].set_ylabel('Target Index')

    # Embedding Similarity After Training
    sns.heatmap(embedding_sim_after, cmap='coolwarm', center=0,
                vmin=-1, vmax=1, ax=axes[2], cbar_kws={'label': 'Similarity'})
    axes[2].set_title('Embedding Similarity (After Training)')
    axes[2].set_xlabel('Target Index')
    axes[2].set_ylabel('Target Index')

    plt.tight_layout()

    # Save image
    filepath = os.path.join(output_dir, 'similarity_heatmaps.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Similarity heatmaps saved to: {filepath}")


def visualize_ligand_profile_comparison(profile_matrix,
                                        embeddings_before,
                                        embeddings_after,
                                        output_dir='UMAP',
                                        embedding_type='molformer_esm',
                                        dataset_name='herb',
                                        clustering_method='eigengap',
                                        exclude_isolated=True, filter_all_isolated=False):
    """
    Generate comparison plots before and after training (isolated nodes excluded version)

    Args:
        profile_matrix: Ligand profile matrix
        embeddings_before: Embeddings before training
        embeddings_after: Embeddings after training
        output_dir: Output directory
        embedding_type: Embedding type ('molformer_esm', 'ecfp_pseaac', 'metapath2vec', or compatible 'pretrained'/'baseline')
        dataset_name: Dataset name
        clustering_method: Clustering method ('eigengap' or 'silhouette')
        exclude_isolated: Whether to exclude isolated nodes
        filter_all_isolated: Whether to filter isolated nodes for ME/EP embeddings as well (MV always filtered)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Ligand Profile Analysis - {dataset_name.upper()} Dataset")
    print(f"Clustering Method: {clustering_method.upper()}")
    print(f"Exclude Isolated Nodes: {'Yes' if exclude_isolated else 'No'}")
    print("=" * 60)

    # ===== Unified Isolated Node Preprocessing =====
    should_filter = False
    filter_reason = ""

    # Determine whether to filter
    embedding_type_lower = embedding_type.lower()

    if embedding_type_lower in ['metapath2vec', 'mv', 'mp2v', 'bipartite', 'bg', 'bipartite_graph']:
        # MV and BG embeddings always filtered (zero vectors meaningless)
        should_filter = True
        if embedding_type_lower in ["bipartite", "bg", "bipartite_graph"]:
            filter_reason = "Bipartite Graph Embeddings (Zero vectors meaningless, always filtered)"
        else:
            filter_reason = "Metapath2Vec Embeddings (Zero vectors meaningless, always filtered)"

    elif filter_all_isolated and embedding_type_lower in ['molformer_esm', 'me', 'molformer', 'esm',
                                                          'ecfp_pseaac', 'ep', 'ecfp', 'pseaac',
                                                          'pretrained', 'baseline']:
        # ME/EP embeddings filtered based on filter_all_isolated parameter
        should_filter = True
        if embedding_type_lower in ['molformer_esm', 'me', 'molformer', 'esm', 'pretrained']:
            filter_reason = "ME Embeddings (filter_all_isolated=True)"
        else:
            filter_reason = "EP Embeddings (filter_all_isolated=True)"

    # Execute filtering
    if should_filter and exclude_isolated:
        print(f"\n=== Isolated Node Preprocessing ===")
        print(f"Reason: {filter_reason}")

        # Detect zero vectors (isolated nodes)
        zero_before = np.all(embeddings_before == 0, axis=1)
        zero_after = np.all(embeddings_after == 0, axis=1)

        # Isolated nodes: Zero vectors either before or after training
        isolated_mask = zero_before | zero_after
        non_isolated_mask = ~isolated_mask

        n_isolated = isolated_mask.sum()
        n_non_isolated = non_isolated_mask.sum()

        print(f"Original Nodes: {len(embeddings_before)}")
        print(f"Isolated Nodes: {n_isolated}")
        print(f"  - Zero Vectors Before Training: {zero_before.sum()}")
        print(f"  - Zero Vectors After Training: {zero_after.sum()}")
        print(f"Non-Isolated Nodes: {n_non_isolated}")

        if n_non_isolated == 0:
            print("Warning: All nodes are isolated!")
            # Do not return None, continue with all-zero data
        else:
            # Filter all data from source
            embeddings_before = embeddings_before[non_isolated_mask]
            embeddings_after = embeddings_after[non_isolated_mask]
            profile_matrix = profile_matrix[non_isolated_mask]

            print(f"\nShape After Filtering:")
            print(f"  Embeddings: {embeddings_before.shape}")
            print(f"  Ligand Profile: {profile_matrix.shape}")

        # Already filtered, no need to filter again
        exclude_isolated = False
    else:
        if not should_filter:
            print(f"\n=== Keeping All Nodes (Including Isolated) ===")
            if embedding_type_lower in ['molformer_esm', 'me', 'molformer', 'esm', 'pretrained']:
                print(f"Reason: ME Embeddings, filter_all_isolated=False")
            elif embedding_type_lower in ['ecfp_pseaac', 'ep', 'ecfp', 'pseaac', 'baseline']:
                print(f"Reason: EP Embeddings, filter_all_isolated=False")
            else:
                print(f"Reason: Filtering not enabled for {embedding_type} embeddings")
            # Set exclude_isolated=False when filtering is disabled
            exclude_isolated = False
    # ===== Preprocessing End =====

    # Compute similarity matrices (exclude isolated nodes)
    result_before = compute_similarity_matrices(
        profile_matrix, embeddings_before, exclude_isolated=exclude_isolated
    )

    result_after = compute_similarity_matrices(
        profile_matrix, embeddings_after, exclude_isolated=exclude_isolated
    )

    if result_before[0] is None or result_after[0] is None:
        print("Error: Unable to compute similarity matrices (possibly all nodes are isolated)")
        return None

    ligand_sim, embedding_sim_before, mask_before = result_before
    _, embedding_sim_after, mask_after = result_after

    # Ensure same mask used before and after
    if exclude_isolated and mask_before is not None and mask_after is not None:
        # Use intersection (nodes non-isolated in both cases)
        common_mask = mask_before & mask_after
        if common_mask.sum() < mask_before.sum() or common_mask.sum() < mask_after.sum():
            print(f"Note: Non-isolated nodes not identical before and after training, using intersection ({common_mask.sum()} nodes)")

            # Refilter
            profile_matrix_filtered = profile_matrix[common_mask, :]
            embeddings_before_filtered = embeddings_before[common_mask, :]
            embeddings_after_filtered = embeddings_after[common_mask, :]

            # Recompute similarities
            profile_norm = normalize(profile_matrix_filtered, axis=1, norm='l2')
            ligand_sim = cosine_similarity(profile_norm)

            embedding_norm_before = normalize(embeddings_before_filtered, axis=1, norm='l2')
            embedding_sim_before = cosine_similarity(embedding_norm_before)

            embedding_norm_after = normalize(embeddings_after_filtered, axis=1, norm='l2')
            embedding_sim_after = cosine_similarity(embedding_norm_after)

            # Update embeddings for subsequent analysis
            embeddings_before = embeddings_before_filtered
            embeddings_after = embeddings_after_filtered

    # 1. Compute Correlation Coefficients
    print("\n### Standard Correlation Analysis (Upper Triangle) ###")
    corr_before, p_before, n_before = calculate_similarity_correlation(
        ligand_sim, embedding_sim_before, exclude_diagonal=True
    )
    corr_after, p_after, n_after = calculate_similarity_correlation(
        ligand_sim, embedding_sim_after, exclude_diagonal=True
    )

    print(f"\nCorrelation Before Training: rho={corr_before:.4f} (p={p_before:.2e})")
    print(f"Correlation After Training: rho={corr_after:.4f} (p={p_after:.2e})")

    improvement = (corr_after - corr_before) / abs(corr_before) * 100 if corr_before != 0 else 0
    print(f"Relative Improvement: {improvement:+.1f}%")

    # 2. Analysis at Different Thresholds
    print("\n### Correlation Analysis at Different Thresholds ###")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    results_before = analyze_similarity_at_thresholds(
        ligand_sim, embedding_sim_before, thresholds
    )
    results_after = analyze_similarity_at_thresholds(
        ligand_sim, embedding_sim_after, thresholds
    )

    print("\nThreshold Analysis Results:")
    print("Threshold | Before rho | After rho | Improvement | Pairs (%) ")
    print("-" * 65)
    for threshold in thresholds:
        before_rho = results_before[threshold]['correlation']
        after_rho = results_after[threshold]['correlation']
        n_pairs = results_before[threshold]['n_pairs']
        percentage = results_before[threshold]['percentage']

        if not np.isnan(before_rho) and not np.isnan(after_rho):
            improvement = (after_rho - before_rho) / abs(before_rho) * 100 if before_rho != 0 else 0
            print(
                f"  {threshold:.1f}    | {before_rho:10.4f} | {after_rho:9.4f} | {improvement:+10.1f}% | {n_pairs:4d} ({percentage:5.1f}%)")
        else:
            print(f"  {threshold:.1f}    |        NaN |       NaN |         N/A | {n_pairs:4d} ({percentage:5.1f}%)")

    # 3. Create Heatmaps (Optional)
    # create_similarity_heatmaps(ligand_sim, embedding_sim_before, embedding_sim_after, output_dir)

    # 4. Clustering Analysis and UMAP Visualization
    print("\n### Clustering Analysis ###")

    # Cluster using Ligand Profile Similarity
    optimal_k, _ = determine_optimal_clusters(
        ligand_sim, method=clustering_method,
        max_clusters=15, min_clusters=2, plot=True
    )

    # Perform clustering
    from sklearn.cluster import SpectralClustering
    clustering = SpectralClustering(n_clusters=optimal_k,
                                    affinity='precomputed',
                                    random_state=42)
    cluster_labels = clustering.fit_predict(ligand_sim)

    # UMAP Visualization
    print("\n### UMAP Visualization ###")
    create_umap_visualization(embeddings_before, embeddings_after,
                              cluster_labels, optimal_k,
                              output_dir, dataset_name, embedding_type)

    # 5. Generate Report
    generate_comparison_report_extended(
        corr_before, corr_after, results_before, results_after,
        output_dir, dataset_name, embedding_type, clustering_method, optimal_k
    )

    # Return results
    results = {
        'corr_before': corr_before,
        'corr_after': corr_after,
        'improvement': improvement,
        'threshold_results_before': results_before,
        'threshold_results_after': results_after,
        'n_clusters': optimal_k,
        'cluster_labels': cluster_labels
    }

    return results


def create_umap_visualization(embedding_2d_before, embedding_2d_after,
                              cluster_labels, n_clusters,
                              output_dir, dataset_name, embedding_type):
    """
    Create UMAP visualization (comparison)
    """
    # UMAP dimensionality reduction
    reducer = umap.UMAP(n_components=2, random_state=42)

    print("Running UMAP dimensionality reduction...")
    embedding_2d_before = reducer.fit_transform(embedding_2d_before)
    embedding_2d_after = reducer.fit_transform(embedding_2d_after)

    # Set palette (Optimized)
    # Define primary colors (7 carefully selected colors)
    primary_colors = [
    '#FF6B6B',  # Red
    '#FF8C42',  # Red-Orange
    '#FF9F1C',  # Orange
    '#FFC300',  # Yellow
    '#2ECC71',  # Yellow-Green
    '#00B894',  # Teal-Green
    '#00C4FF',  # Cyan
    '#1F78FF',  # Blue
    '#A94EFF',  # Blue-Violet
    '#FF4F81',  # Magenta
]

    # Define secondary colors (15 extra colors)
    secondary_colors = ['#E91E63', '#9C27B0', '#673AB7', '#3F51B5',
                        '#00BCD4', '#009688', '#4CAF50', '#8BC34A',
                        '#CDDC39', '#FFEB3B', '#FF5722', '#795548',
                        '#607D8B', '#78909C', '#B0BEC5']

    # Combine all predefined colors
    all_colors = primary_colors + secondary_colors

    # Select colors based on cluster count
    if n_clusters <= len(primary_colors):
        # 7 or less, use primary colors only
        colors = primary_colors[:n_clusters]
    elif n_clusters <= len(all_colors):
        # 8-22, use predefined colors
        colors = all_colors[:n_clusters]
    else:
        # More than 22, generate supplement
        import matplotlib.cm as cm
        cmap_base = cm.get_cmap('tab20')
        extra_needed = n_clusters - len(all_colors)
        extra_colors = [cmap_base(i % 20 / 20) for i in range(extra_needed)]
        colors = all_colors + extra_colors

    cmap = ListedColormap(colors)
    boundaries = np.arange(n_clusters + 1) - 0.5
    norm = BoundaryNorm(boundaries, cmap.N)

    # Create figure with specific layout
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.15, hspace=0)

    ax1 = fig.add_subplot(gs[0, 0], aspect='equal')
    ax2 = fig.add_subplot(gs[0, 1], aspect='equal')
    cbar_ax = fig.add_subplot(gs[0, 2])

    # Helper function to set square limits
    def set_square_limits(ax, data):
        x_min, x_max = data[:, 0].min(), data[:, 0].max()
        y_min, y_max = data[:, 1].min(), data[:, 1].max()

        # Add 10% margin
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1

        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin

        # Compute center and range
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Use larger range to ensure square
        max_range = max(x_range, y_range)

        ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
        ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)

    # Subplot Before Training
    scatter1 = ax1.scatter(embedding_2d_before[:, 0], embedding_2d_before[:, 1],
                           c=cluster_labels, cmap=cmap, norm=norm,
                           s=30, alpha=0.7, edgecolors='white', linewidth=0.5)

    set_square_limits(ax1, embedding_2d_before)
    ax1.set_title('Before Training', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Subplot After Training
    scatter2 = ax2.scatter(embedding_2d_after[:, 0], embedding_2d_after[:, 1],
                           c=cluster_labels, cmap=cmap, norm=norm,
                           s=30, alpha=0.7, edgecolors='white', linewidth=0.5)

    set_square_limits(ax2, embedding_2d_after)
    ax2.set_title('After Training', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add Colorbar
    cbar = plt.colorbar(scatter2, cax=cbar_ax,
                        ticks=np.arange(n_clusters),
                        boundaries=boundaries,
                        drawedges=False)

    # Set tick labels
    cbar.set_ticklabels([f'Cluster {i}' for i in range(n_clusters)])

    # Set tick parameters
    cbar_ax.yaxis.set_tick_params(pad=5, labelsize=9, length=4, width=1, direction='out')

    # Ensure labels centered
    for tick in cbar_ax.yaxis.get_ticklabels():
        tick.set_verticalalignment('center')

    # Adjust layout
    plt.tight_layout()

    # Fine-tune colorbar height to match subplots after tight_layout
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    pos_cbar = cbar_ax.get_position()

    # Compute average position and height
    new_y0 = min(pos1.y0, pos2.y0)
    new_height = max(pos1.y0 + pos1.height, pos2.y0 + pos2.height) - new_y0
    new_x0 = pos2.x0 + pos2.width + 0.02

    # Set colorbar position to match subplot height
    cbar_ax.set_position([new_x0, new_y0, pos_cbar.width, new_height])

    # Save figure
    filename = f'ligand_profile_umap_{dataset_name}_{embedding_type}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    # Print clustering stats
    print(f"\nCluster Size Distribution:")
    for i in range(n_clusters):
        cluster_size = np.sum(cluster_labels == i)
        percentage = cluster_size / len(cluster_labels) * 100
        print(f"  Cluster {i}: {cluster_size} targets ({percentage:.1f}%)")

    # Print results
    embedding_names = {
        'molformer_esm': 'MolFormer+ESM-2',
        'ecfp_pseaac': 'ECFP+PseAAC',
        'metapath2vec': 'Metapath2Vec',
        'pretrained': 'ESM-2+MolFormer',  # Maintain compatibility
        'baseline': 'Baseline'  # Maintain compatibility
    }
    embedding_name = embedding_names.get(embedding_type, embedding_type)
    print(f"\nLigand Profile Analysis Results ({embedding_name} - {dataset_name.upper()}):")
    print(f"  Number of Targets: {len(cluster_labels)}")
    print(f"  Number of Clusters: {n_clusters}")
    print(f"  UMAP plot saved to: {filepath}")


def generate_comparison_report(corr_before, corr_after,
                               results_before, results_after,
                               output_dir, dataset_name, embedding_type):
    """Generate basic comparison analysis report"""

    # Get friendly embedding type name
    embedding_names = {
        'molformer_esm': 'MolFormer+ESM-2',
        'ecfp_pseaac': 'ECFP+PseAAC',
        'metapath2vec': 'Metapath2Vec',
        'pretrained': 'ESM-2+MolFormer',
        'baseline': 'Baseline'
    }
    embedding_display_name = embedding_names.get(embedding_type, embedding_type)

    report_path = os.path.join(output_dir,
                               f'analysis_report_{dataset_name}_{embedding_type}.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Ligand Profile Analysis Report - {dataset_name.upper()}\n")
        f.write(f"Embedding Type: {embedding_display_name}\n")
        f.write("=" * 60 + "\n\n")

        f.write("## Overall Correlation Analysis\n")
        f.write(f"Before Training: rho = {corr_before:.4f}\n")
        f.write(f"After Training: rho = {corr_after:.4f}\n")
        improvement = (corr_after - corr_before) / abs(corr_before) * 100 if corr_before != 0 else 0
        f.write(f"Improvement: {improvement:+.1f}%\n\n")

        f.write("## Threshold Analysis\n")
        f.write("Threshold\tBefore_rho\tAfter_rho\tData Points\n")
        for threshold in results_before.keys():
            before = results_before[threshold]['correlation']
            after = results_after[threshold]['correlation']
            n_pairs = results_before[threshold]['n_pairs']
            f.write(f"{threshold}\t{before:.4f}\t{after:.4f}\t{n_pairs}\n")

        f.write("\n## Analysis Conclusion\n")
        if corr_after > corr_before:
            f.write("✓ Consistency between embedding space and ligand profile functional space improved after training\n")
        else:
            f.write("✗ Consistency between embedding space and ligand profile functional space decreased after training\n")
            f.write("  This may indicate the model learned other features not captured by ligand profiles\n")

    print(f"\nAnalysis report saved to: {report_path}")


def generate_comparison_report_extended(corr_before, corr_after,
                                        results_before, results_after,
                                        output_dir, dataset_name, embedding_type,
                                        clustering_method, n_clusters):
    """Generate extended comparison analysis report with clustering info"""

    # Get friendly embedding type name
    embedding_names = {
        'molformer_esm': 'MolFormer+ESM-2',
        'ecfp_pseaac': 'ECFP+PseAAC',
        'metapath2vec': 'Metapath2Vec',
        'pretrained': 'ESM-2+MolFormer',
        'baseline': 'Baseline'
    }
    embedding_display_name = embedding_names.get(embedding_type, embedding_type)

    report_path = os.path.join(output_dir,
                               f'analysis_report_{dataset_name}_{embedding_type}_{clustering_method}.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Ligand Profile Analysis Report - {dataset_name.upper()}\n")
        f.write(f"Embedding Type: {embedding_display_name}\n")
        f.write(f"Clustering Method: {clustering_method}\n")
        f.write(f"Number of Clusters: {n_clusters}\n")
        f.write("Note: Isolated nodes (zero vectors) excluded\n")
        f.write("=" * 60 + "\n\n")

        f.write("## Overall Correlation Analysis\n")
        f.write(f"Before Training: rho = {corr_before:.4f}\n")
        f.write(f"After Training: rho = {corr_after:.4f}\n")
        improvement = (corr_after - corr_before) / abs(corr_before) * 100 if corr_before != 0 else 0
        f.write(f"Improvement: {improvement:+.1f}%\n\n")

        f.write("## Clustering Info\n")
        f.write(f"Method Used: {clustering_method.upper()}\n")
        f.write(f"Optimal Clusters Found: {n_clusters}\n")
        if clustering_method == 'eigengap':
            f.write("Method Description: Eigengap analysis of normalized Laplacian matrix\n")
        else:
            f.write("Method Description: Maximization of silhouette coefficient\n")
        f.write("\n")

        f.write("## Threshold Analysis\n")
        f.write("Threshold\tBefore_rho\tAfter_rho\tData Points\tPercentage\n")
        for threshold in results_before.keys():
            before = results_before[threshold]['correlation']
            after = results_after[threshold]['correlation']
            n_pairs = results_before[threshold]['n_pairs']
            percentage = results_before[threshold]['percentage']
            f.write(f"{threshold}\t{before:.4f}\t{after:.4f}\t{n_pairs}\t{percentage:.1f}%\n")

        f.write("\n## Analysis Conclusion\n")
        if corr_after > corr_before:
            f.write("✓ Consistency between embedding space and ligand profile functional space improved after training\n")
            f.write(f"  Improvement: {improvement:.1f}%\n")
        else:
            f.write("✗ Consistency between embedding space and ligand profile functional space decreased after training\n")
            f.write(f"  Decrease: {improvement:.1f}%\n")
            f.write("  This may indicate the model learned other features not captured by ligand profiles\n")

    print(f"Extended analysis report saved to: {report_path}")


def find_optimal_clusters_eigengap(similarity_matrix, max_clusters=15,
                                   min_clusters=2, plot=False):
    """
    Find optimal number of clusters using eigengap method

    Args:
        similarity_matrix: Similarity matrix
        max_clusters: Max clusters to try
        min_clusters: Min clusters to try
        plot: Whether to generate eigenvalue plot

    Returns:
        optimal_k: Optimal number of clusters
        eigenvalues: First max_clusters eigenvalues
    """
    print("\nUsing Eigengap method for clustering...")

    # Compute degree matrix
    D = np.diag(np.sum(similarity_matrix, axis=1))

    # Compute normalized Laplacian matrix
    # L = D - W
    L = D - similarity_matrix

    # Avoid division by zero
    D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-10)))

    # L_norm = D^(-1/2) * L * D^(-1/2)
    L_norm = D_sqrt_inv @ L @ D_sqrt_inv

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(L_norm)

    # Sort eigenvalues (ascending)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]

    # Compute eigengap
    gaps = []
    for i in range(min_clusters - 1, min(max_clusters, len(eigenvalues) - 1)):
        gap = eigenvalues[i + 1] - eigenvalues[i]
        gaps.append(gap)

    # Find cluster count corresponding to max gap
    if gaps:
        optimal_idx = np.argmax(gaps)
        optimal_k = optimal_idx + min_clusters
    else:
        optimal_k = min_clusters

    print(f"Eigengap Analysis:")
    for i, gap in enumerate(gaps[:10]):  # Show only first 10
        k = i + min_clusters
        print(f"  k={k}: gap={gap:.6f}")

    print(f"Final selected cluster count: {optimal_k}")

    if plot:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.figure(figsize=(12, 4))

        # Subplot 1: Eigenvalues
        plt.subplot(1, 3, 1)
        n_eigenvalues = min(50, len(eigenvalues))
        plt.plot(range(1, n_eigenvalues + 1), eigenvalues[:n_eigenvalues], 'bo-', markersize=4)
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalues of Normalized Laplacian')
        plt.grid(True, alpha=0.3)

        # Subplot 2: Eigengaps
        if gaps:
            plt.subplot(1, 3, 2)
            k_values = list(range(min_clusters, min_clusters + len(gaps)))
            plt.plot(k_values, gaps, 'ro-', markersize=6)
            plt.axvline(x=optimal_k, color='g', linestyle='--',
                        label=f'Optimal k={optimal_k}')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Eigengap')
            plt.title('Eigengap Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Subplot 3: First 20 Eigenvalues (Zoomed)
        plt.subplot(1, 3, 3)
        n_show = min(20, len(eigenvalues))
        plt.plot(range(1, n_show + 1), eigenvalues[:n_show], 'go-', markersize=6)
        if optimal_k <= n_show:
            plt.axvline(x=optimal_k, color='r', linestyle='--',
                        label=f'Optimal k={optimal_k}')
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.title('First 20 Eigenvalues (Zoomed)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('eigenvalue_analysis.png', dpi=150)
        plt.close()
        print("Eigenvalue analysis plot saved to: eigenvalue_analysis.png")

    return optimal_k, eigenvalues[:max_clusters] if len(eigenvalues) >= max_clusters else eigenvalues


def find_optimal_clusters_silhouette(similarity_matrix, max_clusters=15,
                                     min_clusters=2, plot=False):
    """
    Find optimal number of clusters using Silhouette Score method

    Args:
        similarity_matrix: Similarity matrix
        max_clusters: Max clusters to try
        min_clusters: Min clusters to try
        plot: Whether to generate analysis plot

    Returns:
        optimal_k: Optimal number of clusters
        silhouette_scores: Silhouette scores for each cluster count
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score

    print("\nUsing Silhouette Score method for clustering...")

    # Convert similarity matrix to positive definite matrix
    W = similarity_matrix.copy()
    W = (W + W.T) / 2  # Ensure symmetry
    np.fill_diagonal(W, 1)  # Set diagonal to 1
    W = np.maximum(W, 0)  # Ensure non-negative

    silhouette_scores = []
    k_values = list(range(min_clusters, min(max_clusters + 1, len(W))))

    for k in k_values:
        # Spectral Clustering
        clustering = SpectralClustering(n_clusters=k,
                                        affinity='precomputed',
                                        random_state=42)
        cluster_labels = clustering.fit_predict(W)

        # Calculate Silhouette Score
        # Use 1 - similarity as distance
        distance_matrix = 1 - W
        score = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
        silhouette_scores.append(score)
        print(f"  k={k}: Silhouette Score={score:.4f}")

    # Find optimal k (max silhouette score)
    optimal_idx = np.argmax(silhouette_scores)
    optimal_k = k_values[optimal_idx]
    max_score = silhouette_scores[optimal_idx]

    print(f"Optimal cluster count: k={optimal_k} (Silhouette Score={max_score:.4f})")

    if plot:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.figure(figsize=(10, 5))

        # Plot Silhouette Scores
        plt.subplot(1, 2, 1)
        plt.plot(k_values, silhouette_scores, 'bo-', markersize=8, linewidth=2)
        plt.axvline(x=optimal_k, color='r', linestyle='--',
                    label=f'Optimal k={optimal_k}')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot bar chart for better visualization
        plt.subplot(1, 2, 2)
        colors = ['red' if k == optimal_k else 'blue' for k in k_values]
        plt.bar(k_values, silhouette_scores, color=colors, alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title(f'Optimal k={optimal_k} (score={max_score:.4f})')
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('silhouette_analysis.png', dpi=150)
        plt.close()
        print("Silhouette analysis plot saved to: silhouette_analysis.png")

    return optimal_k, silhouette_scores


def determine_optimal_clusters(similarity_matrix, method='eigengap',
                               max_clusters=15, min_clusters=2, plot=False):
    """
    Determine optimal clusters using specified method

    Args:
        similarity_matrix: Similarity matrix
        method: 'eigengap' or 'silhouette'
        max_clusters: Max clusters to try
        min_clusters: Min clusters to try
        plot: Whether to generate charts

    Returns:
        optimal_k: Optimal number of clusters
        scores: Method-specific scores (Eigenvalues or Silhouette Scores)
    """
    if method == 'eigengap':
        print("\nUsing Eigengap method for clustering...")
        optimal_k, scores = find_optimal_clusters_eigengap(
            similarity_matrix, max_clusters, min_clusters, plot
        )
    elif method == 'silhouette':
        print("\nUsing Silhouette Score method for clustering...")
        optimal_k, scores = find_optimal_clusters_silhouette(
            similarity_matrix, max_clusters, min_clusters, plot
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}. Please use 'eigengap' or 'silhouette'")

    return optimal_k, scores


# Main function example
def run_ligand_profile_analysis(args, hetero_graph, initial_features,
                                model=None, trained_embeddings=None,
                                clustering_method='eigengap', filter_all_isolated=False):
    """
    Main function for Ligand Profile Analysis (Isolated nodes excluded version)

    Args:
        args: Argument object containing dataset and settings
        hetero_graph: Heterogeneous graph object
        initial_features: Initial feature embeddings
        model: Trained model (optional)
        trained_embeddings: Trained embeddings (optional)
        clustering_method: 'eigengap' or 'silhouette' (default: 'eigengap')
        filter_all_isolated: Whether to filter isolated nodes for ME/EP embeddings too (MV always filtered)
    """

    #num_targets = hetero_graph.num_nodes('target')
    #num_ingredients = hetero_graph.num_nodes('ingredient')
    if isinstance(initial_features, dict) and 'pretrained' in initial_features:
        num_targets = initial_features['pretrained']['target'].shape[0]
        num_ingredients = initial_features['pretrained']['ingredient'].shape[0]
    else:
        num_targets = initial_features['target'].shape[0]
        num_ingredients = initial_features['ingredient'].shape[0]

    print(f"\n{'=' * 60}")
    print(f"Ligand Profile Analysis - {args.dataset.upper()}")
    print(f"Clustering Method: {clustering_method.upper()}")
    print(f"Mode: {'Exclude Isolated Nodes' if filter_all_isolated else 'Keep Isolated Nodes'}")
    print(f"{'=' * 60}")
    print(f"Number of Targets: {num_targets}")
    print(f"Number of Ingredients/Drugs: {num_ingredients}")

    # Get edge file path
    edge_file = get_edge_file_path(args.dataset)

    if not os.path.exists(edge_file):
        print(f"Error: Edge file {edge_file} not found")
        return None

    # Construct ligand profile
    profile_matrix = construct_ligand_profile(
        edge_file, num_targets, num_ingredients, use_tfidf=True
    )
    print(f"Ligand Profile Matrix Shape: {profile_matrix.shape}")

    # Get initial embeddings
    if isinstance(initial_features, dict) and 'pretrained' in initial_features:
        initial_target_embeddings = initial_features['pretrained']['target'].cpu().numpy()
    else:
        initial_target_embeddings = initial_features['target'].cpu().numpy()

    # If trained embeddings are provided
    if trained_embeddings is not None:
        trained_target_embeddings = trained_embeddings['target'].cpu().numpy()

        # Determine embedding type name based on vector type
        if hasattr(args, 'herb_use_baseline'):
            if args.herb_use_baseline == 'ME':
                embedding_type = 'molformer_esm'
            elif args.herb_use_baseline == 'EP':
                embedding_type = 'ecfp_pseaac'
            elif args.herb_use_baseline == 'MV':
                embedding_type = 'metapath2vec'
            elif args.herb_use_baseline == "BG":
                embedding_type = "bipartite_graph"
            else:
                # Compatible with old boolean values
                embedding_type = 'baseline' if args.herb_use_baseline else 'pretrained'
        else:
            embedding_type = 'pretrained'  # Default

        # Run analysis using selected clustering method (excluding isolated nodes)
        results = visualize_ligand_profile_comparison(
            profile_matrix,
            initial_target_embeddings,
            trained_target_embeddings,
            output_dir='UMAP',
            embedding_type=embedding_type,
            dataset_name=args.dataset,
            clustering_method=clustering_method,
            exclude_isolated=True,  # Exclude isolated nodes
            filter_all_isolated=filter_all_isolated  # Control whether ME/EP is filtered
        )

        return results
    else:
        print("Trained embeddings not provided, skipping analysis")
        return None


def get_edge_file_path(dataset_name):
    """Get edge file path for dataset"""
    dataset_paths = {
        'herb': os.path.join("data", "ingredient_target_edges.txt"),
        'drugbank': os.path.join("drugbank", "drug_target_edges.txt"),
        'davis': os.path.join("davis", "drug_target_edges.txt"),
        'bindingdb': os.path.join("bindingdb", "drug_target_edges.txt")
    }

    if dataset_name not in dataset_paths:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset_paths[dataset_name]


if __name__ == "__main__":
    # Test code
    print("Ligand Profile Analysis Tool (Isolated Nodes Excluded Version)")
    print("Key Features:")
    print("1. Automatically detects and excludes isolated nodes (zero vectors)")
    print("2. Uses upper triangle matrix for correlation calculation to avoid redundancy")
    print("3. Provides correlation analysis at different thresholds")
    print("4. Enhanced visualization")
    print("5. Generates detailed analysis report")