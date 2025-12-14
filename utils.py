import torch
import dgl
import os
import pickle as pkl
import random
import numpy as np
import pandas as pd
from numpy.linalg import norm
import torch.nn.functional as F

def add_gaussian_noise_to_features(initial_features, noise_std, noise_types, device='cuda'):
    """
    Add Gaussian noise to features of specified node types

    Args:
        initial_features: Initial feature dictionary {'pretrained': {'ingredient': tensor, 'target': tensor, ...}}
        noise_std: Standard deviation of Gaussian noise
        noise_types: List of node types to add noise to, e.g. ['ingredient', 'target']
        device: Device

    Returns:
        noisy_features: Feature dictionary after adding noise (same structure)
    """
    if noise_std == 0:
        return initial_features

    print(f"\nAdding Gaussian noise to features (std={noise_std}, types={noise_types})")

    # Deep copy to avoid modifying original data
    noisy_features = {'pretrained': {}}

    for ntype, features in initial_features['pretrained'].items():
        if ntype in noise_types:
            # Add Gaussian noise
            noise = torch.randn_like(features) * noise_std
            noisy_features['pretrained'][ntype] = features + noise

            # Print statistics
            original_norm = torch.norm(features).item()
            noise_norm = torch.norm(noise).item()
            print(f"  {ntype}: Original Feature Norm={original_norm:.2f}, "
                  f"Noise Norm={noise_norm:.2f}, "
                  f"SNR={original_norm / noise_norm:.2f}")
        else:
            # Directly copy node types without noise
            noisy_features['pretrained'][ntype] = features
            print(f"  {ntype}: Keeping original features (no noise)")

    return noisy_features
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    dgl.seed(seed)


def read_txt(file):
    res_list = []
    with open(file, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
            parts = line.split()  # split() automatically handles spaces and tabs
            if len(parts) >= 2:
                res_list.append([int(parts[0]), int(parts[1])])
    return res_list


def process_data(vector_type='ME', baseline_dims=None):

    base_path = "data/"

    # Edge relationship file paths (unchanged)
    ingredient_target_edge_file = "ingredient_target_edges.txt"
    target_target_file = "target_similarity.txt"
    ingredient_similarity_file = "ingredient_similarity.txt"
    herb_ingredient_edge_file = "herb_ingredient_edges.txt"

    # ============= Select vector file based on vector type =============
    if vector_type == 'EP':
        # Use baseline vectors (ECFP + PseAAC)
        ingredient_embeddings_file = "pre-compare/drug_ecfp_vectors.npy"
        target_embeddings_file = "pre-compare/target_pseaac_vectors.npy"
        print(f">>> Using Baseline Vectors (EP):")
        print(f"    Ingredient: ECFP (Expected dim: {baseline_dims[0] if baseline_dims else 'N/A'})")
        print(f"    Target: PseAAC (Expected dim: {baseline_dims[1] if baseline_dims else 'N/A'})")
    elif vector_type == 'MV':
        # Use Metapath2Vec vectors
        ingredient_embeddings_file = "Metapath2Vec/ingredient_embeddings.npy"
        target_embeddings_file = "Metapath2Vec/target_embeddings.npy"
        print(">>> Using Metapath2Vec Vectors (MV)")
    elif vector_type == "BG":
        # Use Bipartite Graph vectors
        ingredient_embeddings_file = "bipartitegraph/ingredient_embeddings.npy"
        target_embeddings_file = "bipartitegraph/target_embeddings.npy"
        print(">>> Using Bipartite Graph Vectors (BG)")
        print("    Structured embeddings based on bipartite graph projection")
    else:  # ME (Default)
        # Use pretrained vectors
        ingredient_embeddings_file = "smiles_vectors_molformer.npy"
        target_embeddings_file = "protein_embeddings.npy"
        print(">>> Using Pretrained Vectors (ME): MolFormer + ESM-2")

    # Herb vector file remains unchanged
    herb_embeddings_file = "herb_vectors.npy"
    # ==============================================

    # Read training data
    existing_edges = read_txt(os.path.join(base_path, ingredient_target_edge_file))
    ingredient_similarity = read_txt(os.path.join(base_path, ingredient_similarity_file))
    target_similarity = read_txt(os.path.join(base_path, target_target_file))
    herb_ingredient_edges = read_txt(os.path.join(base_path, herb_ingredient_edge_file))

    # Load embedding files
    try:
        ingredient_embeddings = np.load(os.path.join(base_path, ingredient_embeddings_file))
        target_embeddings = np.load(os.path.join(base_path, target_embeddings_file))
        herb_embeddings = np.load(os.path.join(base_path, herb_embeddings_file))
    except FileNotFoundError as e:
        if vector_type == 'EP':
            print(f"Error: Baseline vector files not found. Please ensure the following files exist:")
            print(f"  - {os.path.join(base_path, ingredient_embeddings_file)}")
            print(f"  - {os.path.join(base_path, target_embeddings_file)}")
        elif vector_type == 'MV':
            print(f"Error: Metapath2Vec vector files not found. Please ensure the following files exist:")
            print(f"  - {os.path.join(base_path, ingredient_embeddings_file)}")
            print(f"  - {os.path.join(base_path, target_embeddings_file)}")
        elif vector_type == "BG":
            print(f"Error: Bipartite Graph vector files not found. Please ensure the following files exist:")
            print(f"  - {os.path.join(base_path, ingredient_embeddings_file)}")
            print(f"  - {os.path.join(base_path, target_embeddings_file)}")
        raise e

    print(f"Herb Embedding Shape: {herb_embeddings.shape}")
    print(f"Ingredient Embedding Shape: {ingredient_embeddings.shape}")
    print(f"Target Embedding Shape: {target_embeddings.shape}")

    # ============= Validate baseline vector dimensions =============
    if vector_type == 'EP' and baseline_dims:
        if ingredient_embeddings.shape[1] != baseline_dims[0]:
            print(f"Warning: Ingredient vector dimension mismatch!")
            print(f"  Expected: {baseline_dims[0]}")
            print(f"  Actual: {ingredient_embeddings.shape[1]}")
            print("Continuing with actual dimensions...")

        if target_embeddings.shape[1] != baseline_dims[1]:
            print(f"Warning: Target vector dimension mismatch!")
            print(f"  Expected: {baseline_dims[1]}")
            print(f"  Actual: {target_embeddings.shape[1]}")
            print("Continuing with actual dimensions...")
    elif vector_type == 'MV':
        # Metapath2Vec vectors, print actual dimensions
        print(f"Metapath2Vec Vector Dimensions:")
        print(f"  Ingredient: {ingredient_embeddings.shape[1]}")
        print(f"  Target: {target_embeddings.shape[1]}")
    elif vector_type == "BG":
        # Bipartite Graph vectors, print actual dimensions
        print(f"Bipartite Graph Vector Dimensions:")
        print(f"  Ingredient: {ingredient_embeddings.shape[1]}")
        print(f"  Target: {target_embeddings.shape[1]}")
    # =========================================

    # Convert to torch tensors
    herb_embeddings = torch.FloatTensor(herb_embeddings)
    ingredient_embeddings = torch.FloatTensor(ingredient_embeddings)
    target_embeddings = torch.FloatTensor(target_embeddings)

    # Build initial feature dictionary (pretrained embeddings)
    initial_features = {
        'pretrained': {
            'herb': herb_embeddings,
            'ingredient': ingredient_embeddings,
            'target': target_embeddings
        }
    }

    # Convert edge data to DataFrame
    edges = pd.DataFrame(existing_edges, columns=['source', 'target'])
    edges = edges.set_index(edges.index.astype(str))

    is_edges = pd.DataFrame(ingredient_similarity, columns=['source', 'target'])
    is_edges = is_edges.set_index(is_edges.index.astype(str))

    ts_edges = pd.DataFrame(target_similarity, columns=['source', 'target'])
    ts_edges = ts_edges.set_index(ts_edges.index.astype(str))

    hi_edges = pd.DataFrame(herb_ingredient_edges, columns=['source', 'target'])
    hi_edges = hi_edges.set_index(hi_edges.index.astype(str))

    return edges, is_edges, ts_edges, hi_edges, initial_features


def process_drugbank_data():
    """
    Process DrugBank dataset (Drug-Target network)

    Returns:
        edges: drug-target edges
        is_edges: drug similarity edges
        ts_edges: target similarity edges
        initial_features: contains pretrained embeddings
    """
    base_path = "drugbank/"

    # File paths
    drug_target_edge_file = "drug_target_edges.txt"
    drug_similarity_file = "smile_similarity.txt"
    target_similarity_file = "protein_similarity.txt"

    # Embedding files (pretrained features)
    drug_embeddings_file = "smiles_vectors_molformer.npy"
    target_embeddings_file = "protein_embeddings.npy"

    # Read edge relationships
    drug_target_edges = read_txt(os.path.join(base_path, drug_target_edge_file))
    drug_similarity = read_txt(os.path.join(base_path, drug_similarity_file))
    target_similarity = read_txt(os.path.join(base_path, target_similarity_file))

    # Load embeddings (pretrained features)
    drug_embeddings = np.load(os.path.join(base_path, drug_embeddings_file))
    target_embeddings = np.load(os.path.join(base_path, target_embeddings_file))

    print(f"Drug Embedding Shape: {drug_embeddings.shape}")
    print(f"Target Embedding Shape: {target_embeddings.shape}")
    print(f"Drug-Target Edges: {len(drug_target_edges)}")
    print(f"Drug Similarity Edges: {len(drug_similarity)}")
    print(f"Target Similarity Edges: {len(target_similarity)}")

    # Convert to torch tensors
    drug_embeddings = torch.FloatTensor(drug_embeddings)
    target_embeddings = torch.FloatTensor(target_embeddings)

    # Build initial feature dictionary (pretrained embeddings) - Note: drug mapped to ingredient for compatibility
    initial_features = {
        'pretrained': {
            'ingredient': drug_embeddings,  # Drug in DrugBank corresponds to ingredient in original system
            'target': target_embeddings
        }
    }

    # Convert to DataFrame
    edges = pd.DataFrame(drug_target_edges, columns=['source', 'target'])
    edges = edges.set_index(edges.index.astype(str))

    is_edges = pd.DataFrame(drug_similarity, columns=['source', 'target'])
    is_edges = is_edges.set_index(is_edges.index.astype(str))

    ts_edges = pd.DataFrame(target_similarity, columns=['source', 'target'])
    ts_edges = ts_edges.set_index(ts_edges.index.astype(str))

    return edges, is_edges, ts_edges, initial_features


def process_davis_data():
    """
    Process Davis dataset (Drug-Target network)

    Returns:
        edges: drug-target edges
        is_edges: drug similarity edges
        ts_edges: target similarity edges
        initial_features: contains pretrained embeddings
    """
    base_path = "davis/"

    # File paths
    drug_target_edge_file = "drug_target_edges.txt"
    drug_similarity_file = "smile_similarity.txt"
    target_similarity_file = "protein_similarity.txt"

    # Embedding files (pretrained features)
    drug_embeddings_file = "smiles_vectors_molformer.npy"
    target_embeddings_file = "protein_embeddings.npy"

    # Read edge relationships
    drug_target_edges = read_txt(os.path.join(base_path, drug_target_edge_file))
    drug_similarity = read_txt(os.path.join(base_path, drug_similarity_file))
    target_similarity = read_txt(os.path.join(base_path, target_similarity_file))

    # Load embeddings (pretrained features)
    drug_embeddings = np.load(os.path.join(base_path, drug_embeddings_file))
    target_embeddings = np.load(os.path.join(base_path, target_embeddings_file))

    print(f"Davis Dataset - Drug Embedding Shape: {drug_embeddings.shape}")
    print(f"Davis Dataset - Target Embedding Shape: {target_embeddings.shape}")
    print(f"Davis Dataset - Drug-Target Edges: {len(drug_target_edges)}")
    print(f"Davis Dataset - Drug Similarity Edges: {len(drug_similarity)}")
    print(f"Davis Dataset - Target Similarity Edges: {len(target_similarity)}")

    # Convert to torch tensors
    drug_embeddings = torch.FloatTensor(drug_embeddings)
    target_embeddings = torch.FloatTensor(target_embeddings)

    # Build initial feature dictionary (pretrained embeddings) - Note: drug mapped to ingredient for compatibility
    initial_features = {
        'pretrained': {
            'ingredient': drug_embeddings,  # Drug in Davis corresponds to ingredient in original system
            'target': target_embeddings
        }
    }

    # Convert to DataFrame
    edges = pd.DataFrame(drug_target_edges, columns=['source', 'target'])
    edges = edges.set_index(edges.index.astype(str))

    is_edges = pd.DataFrame(drug_similarity, columns=['source', 'target'])
    is_edges = is_edges.set_index(is_edges.index.astype(str))

    ts_edges = pd.DataFrame(target_similarity, columns=['source', 'target'])
    ts_edges = ts_edges.set_index(ts_edges.index.astype(str))

    return edges, is_edges, ts_edges, initial_features


def process_bindingdb_data():
    """
    Process BindingDB dataset, including test set (if exists)

    Returns:
        If test set exists: (edges, is_edges, ts_edges, initial_features, test_edges)
        If no test set: (edges, is_edges, ts_edges, initial_features)
    """
    base_path = "bindingDB/"
    test_path = os.path.join(base_path, "test/")

    # File paths
    drug_target_edge_file = "drug_target_edges.txt"
    drug_similarity_file = "smile_similarity.txt"
    target_similarity_file = "protein_similarity.txt"

    # Embedding files (in main directory, contains all nodes) - pretrained features
    drug_embeddings_file = "smiles_vectors_molformer.npy"
    target_embeddings_file = "protein_embeddings.npy"

    # Read training set edge relationships
    drug_target_edges = read_txt(os.path.join(base_path, drug_target_edge_file))
    drug_similarity = read_txt(os.path.join(base_path, drug_similarity_file))
    target_similarity = read_txt(os.path.join(base_path, target_similarity_file))

    # Load embeddings (contains all nodes, shared by training and test sets) - pretrained features
    drug_embeddings = np.load(os.path.join(base_path, drug_embeddings_file))
    target_embeddings = np.load(os.path.join(base_path, target_embeddings_file))

    print(f"BindingDB Dataset - Drug Embedding Shape: {drug_embeddings.shape}")
    print(f"BindingDB Dataset - Target Embedding Shape: {target_embeddings.shape}")
    print(f"BindingDB Dataset - Training Set Drug-Target Edges: {len(drug_target_edges)}")
    print(f"BindingDB Dataset - Drug Similarity Edges: {len(drug_similarity)}")
    print(f"BindingDB Dataset - Target Similarity Edges: {len(target_similarity)}")

    # Check if test set exists
    test_edges = None
    if os.path.exists(test_path):
        test_edge_file = os.path.join(test_path, "drug_target_edges.txt")
        if os.path.exists(test_edge_file):
            test_edges_list = read_txt(test_edge_file)
            print(f"BindingDB Dataset - Found Independent Test Set: {len(test_edges_list)} edges")

            # Convert to DataFrame format
            test_edges = pd.DataFrame(test_edges_list, columns=['source', 'target'])
            test_edges = test_edges.set_index(test_edges.index.astype(str))
        else:
            print("BindingDB Dataset - Test set file not found")
    else:
        print("BindingDB Dataset - Test subdirectory not found")

    # Convert to torch tensors
    drug_embeddings = torch.FloatTensor(drug_embeddings)
    target_embeddings = torch.FloatTensor(target_embeddings)

    # Build initial feature dictionary (pretrained embeddings, contains features for all nodes)
    initial_features = {
        'pretrained': {
            'ingredient': drug_embeddings,  # Contains all drugs from training and test sets
            'target': target_embeddings  # Contains all targets from training and test sets
        }
    }

    # Convert training set to DataFrame
    edges = pd.DataFrame(drug_target_edges, columns=['source', 'target'])
    edges = edges.set_index(edges.index.astype(str))

    is_edges = pd.DataFrame(drug_similarity, columns=['source', 'target'])
    is_edges = is_edges.set_index(is_edges.index.astype(str))

    ts_edges = pd.DataFrame(target_similarity, columns=['source', 'target'])
    ts_edges = ts_edges.set_index(ts_edges.index.astype(str))

    if test_edges is not None:
        return edges, is_edges, ts_edges, initial_features, test_edges

    return edges, is_edges, ts_edges, initial_features


def build_graph(args, edges, is_edges, ts_edges, hi_edges, initial_features, device, num_nodes_dict=None):
    """
    Build heterogeneous graph (using pretrained embeddings)

    Args:
        args: Parameter config
        edges: Main edges (ingredient-target)
        is_edges: Ingredient similarity edges
        ts_edges: Target similarity edges
        hi_edges: Herb-ingredient edges
        initial_features: Initial node features (pretrained embeddings)
        device: Device
        num_nodes_dict: Optional, explicitly specify node counts {'ingredient': N, 'target': M}
    """
    os.environ['DGLBACKEND'] = 'pytorch'
    train_edges_tensor = torch.from_numpy(edges.values)

    # Determine whether to include herb nodes based on remove_herb parameter
    if args.remove_herb or hi_edges is None:
        # Remove herb nodes or DrugBank/Davis/BindingDB dataset: only include ingredients and targets
        rel_list = [
            ('ingredient', 'it', 'target'),
            ('target', 'ti', 'ingredient'),
            ('ingredient', 'is', 'ingredient'),
            ('target', 'ts', 'target')
        ]

        graph_data = {
            rel_list[0]: (train_edges_tensor[:, 0], train_edges_tensor[:, 1]),
            rel_list[1]: (train_edges_tensor[:, 1], train_edges_tensor[:, 0])
        }

        # Add similarity edges based on graph_struct
        # graph_struct=1 or 3: Add ingredient similarity
        if args.graph_struct in [1, 3]:
            is_edges_tensor = torch.from_numpy(is_edges.values)
            graph_data[rel_list[2]] = (torch.cat([is_edges_tensor[:, 0], is_edges_tensor[:, 1]]),
                                       torch.cat([is_edges_tensor[:, 1], is_edges_tensor[:, 0]]))

        # graph_struct=2 or 3: Add target similarity
        if args.graph_struct in [2, 3]:
            ts_edges_tensor = torch.from_numpy(ts_edges.values)
            graph_data[rel_list[3]] = (torch.cat([ts_edges_tensor[:, 0], ts_edges_tensor[:, 1]]),
                                       torch.cat([ts_edges_tensor[:, 1], ts_edges_tensor[:, 0]]))
    else:
        # Keep herb nodes: include all node types
        rel_list = [
            ('ingredient', 'it', 'target'),
            ('target', 'ti', 'ingredient'),
            ('ingredient', 'is', 'ingredient'),
            ('target', 'ts', 'target'),
            ('herb', 'hi', 'ingredient'),
            ('ingredient', 'ih', 'herb')
        ]

        graph_data = {
            rel_list[0]: (train_edges_tensor[:, 0], train_edges_tensor[:, 1]),
            rel_list[1]: (train_edges_tensor[:, 1], train_edges_tensor[:, 0])
        }

        # Add herb-ingredient edges
        hi_edges_tensor = torch.from_numpy(hi_edges.values)
        graph_data[rel_list[4]] = (hi_edges_tensor[:, 0], hi_edges_tensor[:, 1])
        graph_data[rel_list[5]] = (hi_edges_tensor[:, 1], hi_edges_tensor[:, 0])

        # Add similarity edges based on graph_struct
        # graph_struct=1 or 3: Add ingredient similarity
        if args.graph_struct in [1, 3]:
            is_edges_tensor = torch.from_numpy(is_edges.values)
            graph_data[rel_list[2]] = (torch.cat([is_edges_tensor[:, 0], is_edges_tensor[:, 1]]),
                                       torch.cat([is_edges_tensor[:, 1], is_edges_tensor[:, 0]]))

        # graph_struct=2 or 3: Add target similarity
        if args.graph_struct in [2, 3]:
            ts_edges_tensor = torch.from_numpy(ts_edges.values)
            graph_data[rel_list[3]] = (torch.cat([ts_edges_tensor[:, 0], ts_edges_tensor[:, 1]]),
                                       torch.cat([ts_edges_tensor[:, 1], ts_edges_tensor[:, 0]]))

    # Create heterogeneous graph
    if num_nodes_dict:
        # If node counts are specified, use them (for BindingDB and other special cases)
        print(f"Using specified node counts: {num_nodes_dict}")
        hetero_graph = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
    else:
        # Otherwise let DGL automatically infer node counts
        hetero_graph = dgl.heterograph(graph_data)

    # Set node features (pretrained embeddings)
    if args.remove_herb or hi_edges is None:
        # Remove herb or DrugBank/Davis/BindingDB: only set features for ingredients and targets
        node_features = {
            'ingredient': initial_features['pretrained']['ingredient'],
            'target': initial_features['pretrained']['target']
        }
    else:
        # Include features for all node types
        node_features = initial_features['pretrained']

    hetero_graph.ndata['features'] = node_features
    hetero_graph = hetero_graph.to(device)

    print(f"\nConstructed Heterogeneous Graph Info:")
    print(f"Remove Herb Nodes: {'Yes' if args.remove_herb else 'No'}")
    print(f"Similarity Network Config: ", end='')
    if args.graph_struct == 0:
        print("No similarity network")
    elif args.graph_struct == 1:
        print("Ingredient similarity only")
    elif args.graph_struct == 2:
        print("Target similarity only")
    elif args.graph_struct == 3:
        print("Ingredient + Target similarity")

    print(f"\nNode Statistics:")
    for ntype in hetero_graph.ntypes:
        print(f"  {ntype}: {hetero_graph.num_nodes(ntype)} nodes")
    print(f"\nEdge Statistics:")
    for etype in hetero_graph.etypes:
        print(f"  {etype}: {hetero_graph.num_edges(etype)} edges")

    return hetero_graph, rel_list


def compute_loss1(pos_score, neg_score, etype):
    n_edges = pos_score[etype].shape[0]

    if n_edges == 0 or neg_score[etype].numel() == 0:
        return torch.tensor(0.0, dtype=pos_score[etype].dtype, device=pos_score[etype].device, requires_grad=True)

    return (1 - pos_score[etype].unsqueeze(1) + neg_score[etype].view(n_edges, -1)).clamp(min=0).mean()


def compute_loss(pos_score, neg_score, etype):
    """Binary Cross Entropy Loss - Recommended for link prediction"""
    pos_scores = pos_score[etype]
    neg_scores = neg_score[etype]

    # Create labels
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)

    # Combine scores and labels
    all_scores = torch.cat([pos_scores, neg_scores])
    all_labels = torch.cat([pos_labels, neg_labels])

    # Apply sigmoid and compute BCE loss
    return F.binary_cross_entropy_with_logits(all_scores, all_labels)


def cos_sim(a, b):
    # Compute norm
    norm_a = norm(a, axis=1)
    norm_b = norm(b, axis=1)

    # Avoid division by zero
    norm_a = np.where(norm_a == 0, 1e-8, norm_a)
    norm_b = np.where(norm_b == 0, 1e-8, norm_b)

    # Compute cosine similarity
    cos_sim_values = np.sum(a * b, axis=1) / (norm_a * norm_b)

    # Handle potential NaN or infinite values
    cos_sim_values = np.nan_to_num(cos_sim_values, nan=0.0, posinf=1.0, neginf=-1.0)

    # Ensure values are within [-1, 1] range
    cos_sim_values = np.clip(cos_sim_values, -1.0, 1.0)

    return cos_sim_values


def remove_unseen_nodes(node_type, graph, unseen_nodes_to_remove):
    unseen_nodes = graph.nodes(node_type).cpu().numpy()
    src, dst = graph.edges(etype='it')
    mask = np.isin(unseen_nodes, unseen_nodes_to_remove)
    nodes_to_remove = unseen_nodes[mask]
    nodes_to_remove = torch.tensor(nodes_to_remove, device=graph.device)
    remove_src = []
    remove_dst = []
    if node_type == 'ingredient':
        for i in range(len(src)):
            if src[i] in nodes_to_remove:
                remove_src.append(src[i])
                remove_dst.append(dst[i])
    elif node_type == 'target':
        for i in range(len(dst)):
            if dst[i] in nodes_to_remove:
                remove_src.append(src[i])
                remove_dst.append(dst[i])

    graph = dgl.remove_nodes(graph, nodes_to_remove, ntype=node_type)
    return graph, remove_src, remove_dst


def negative_sampling(unseen_setting, edges_to_keep_src, edges_to_keep_dst, num_targets, device='cuda', batch_size=100):
    neg_edges_src = []
    neg_edges_dst = []

    dst_set = set(edges_to_keep_dst.cpu().numpy())
    edges_to_keep_src = edges_to_keep_src.to(device)

    if unseen_setting == 1:
        for src in edges_to_keep_src:
            # Batch generate candidate negative samples on GPU
            candidates = torch.randint(0, num_targets, (batch_size,), device=device)

            # Filter out existing positive target nodes
            valid_neg_dst = candidates[~torch.isin(candidates, edges_to_keep_dst.to(device))]

            if len(valid_neg_dst) > 0:
                neg_dst = valid_neg_dst[0]
            else:
                neg_dst = torch.randint(0, num_targets, (1,), device=device).item()

            neg_edges_src.append(src)
            neg_edges_dst.append(neg_dst)

    elif unseen_setting == 2:
        for dst in edges_to_keep_dst:
            # Batch generate candidate negative samples on GPU
            candidates = torch.randint(0, num_targets, (batch_size,), device=device)

            # Filter out existing positive target nodes
            valid_neg_src = candidates[~torch.isin(candidates, edges_to_keep_src.to(device))]

            if len(valid_neg_src) > 0:
                neg_src = valid_neg_src[0]
            else:
                neg_src = torch.randint(0, num_targets, (1,), device=device).item()

            neg_edges_src.append(neg_src)
            neg_edges_dst.append(dst)

    return torch.tensor(neg_edges_src, device=device), torch.tensor(neg_edges_dst, device=device)


def compute_node_similarity(features1, features2, top_k=5):
    """Calculate cosine similarity between nodes and return top-k similar nodes"""
    # Normalize features
    features1_norm = features1 / (torch.norm(features1, dim=1, keepdim=True) + 1e-8)
    features2_norm = features2 / (torch.norm(features2, dim=1, keepdim=True) + 1e-8)

    # Compute similarity matrix
    similarity = torch.mm(features1_norm, features2_norm.t())

    # Get top-k similar nodes
    top_k_values, top_k_indices = torch.topk(similarity, k=min(top_k, similarity.size(1)), dim=1)

    return top_k_indices, top_k_values


def add_redundant_noise(hetero_graph, noise_ratio, noise_dis, noise_etype='it', device='cuda', verbose=True):
    """
    Vectorized noise generation - Optimized version
    noise_dis: [random, block community, feature perturbation induced, multi-modal mixed, reserved]
    """
    assert len(noise_dis) == 5, "noise_dis must contain 5 elements"

    if verbose:
        print(f"\nAdding noise (ratio={noise_ratio}, etype={noise_etype})")
        print(f"Noise Distribution: Random={noise_dis[0]:.2f}, Block Community={noise_dis[1]:.2f}, "
              f"Feature Perturbation={noise_dis[2]:.2f}, Multi-modal Mixed={noise_dis[3]:.2f}")

    # Determine source and target node types based on edge type
    if noise_etype == 'it':
        src_ntype = 'ingredient'
        dst_ntype = 'target'
    elif noise_etype == 'hi':
        src_ntype = 'herb'
        dst_ntype = 'ingredient'
    else:
        raise ValueError(f"Unsupported edge type: {noise_etype}")

    # Get original edges and node counts
    src_orig, dst_orig = hetero_graph.edges(etype=noise_etype)
    num_src = hetero_graph.num_nodes(src_ntype)
    num_dst = hetero_graph.num_nodes(dst_ntype)
    num_orig_edges = len(src_orig)

    # Convert existing edges to set for fast duplicate checking
    existing_edges = set(zip(src_orig.cpu().numpy(), dst_orig.cpu().numpy()))

    # Get node features (for feature perturbation noise)
    src_features = hetero_graph.nodes[src_ntype].data.get('features', None)
    dst_features = hetero_graph.nodes[dst_ntype].data.get('features', None)

    # Estimate total noise edges
    total_noise_edges = int(num_orig_edges * noise_ratio)

    # Calculate edge count for each noise type
    noise_counts = np.array(noise_dis) / sum(noise_dis)
    noise_per_type = (noise_counts * total_noise_edges).astype(int)

    all_noise_src = []
    all_noise_dst = []

    # ========== Strategy 1: Random Noise (Retained) ==========
    if noise_per_type[0] > 0:
        num_random_edges = noise_per_type[0]
        random_src = torch.randint(0, num_src, (num_random_edges,), device=device)
        random_dst = torch.randint(0, num_dst, (num_random_edges,), device=device)
        all_noise_src.append(random_src)
        all_noise_dst.append(random_dst)

    # ========== Strategy 2: Block Community (Moved from 5th to 2nd position) ==========
    if noise_per_type[1] > 0:
        # Module control switches: [base community, forward boundary pollution, reverse boundary pollution, remote bridging]
        module_switches = [1, 1, 1, 1]

        num_communities = min(15, min(num_src, num_dst) // 8)

        if num_communities > 0:
            edges_per_community_base = noise_per_type[1] // num_communities

            for c in range(num_communities):
                size_variation = 1.0 + 0.5 * np.sin(c * np.pi / num_communities)
                src_size = int((num_src / num_communities) * size_variation)
                dst_size = int((num_dst / num_communities) * size_variation)

                src_start = (c * num_src) // num_communities
                src_end = min(src_start + src_size, num_src)
                dst_start = (c * num_dst) // num_communities
                dst_end = min(dst_start + dst_size, num_dst)

                # Base community dense connection
                if module_switches[0] == 1:
                    dense_factor = 4
                    dense_edges = min(edges_per_community_base * dense_factor,
                                      (src_end - src_start) * (dst_end - dst_start) // 1)

                    src_indices = torch.randint(src_start, src_end, (dense_edges,), device=device)
                    dst_indices = torch.randint(dst_start, dst_end, (dense_edges,), device=device)

                    all_noise_src.append(src_indices)
                    all_noise_dst.append(dst_indices)

                # Forward boundary pollution
                if module_switches[1] == 1 and c < num_communities - 1:
                    boundary_noise = edges_per_community_base // 2
                    next_dst_start = ((c + 1) * num_dst) // num_communities
                    next_dst_end = min(((c + 2) * num_dst) // num_communities, num_dst)

                    boundary_src = torch.randint(src_start, src_end, (boundary_noise,), device=device)
                    boundary_dst = torch.randint(next_dst_start, next_dst_end, (boundary_noise,), device=device)
                    all_noise_src.append(boundary_src)
                    all_noise_dst.append(boundary_dst)

                # Reverse boundary pollution
                if module_switches[2] == 1 and c > 0 and c < num_communities - 1:
                    boundary_noise = edges_per_community_base // 2
                    prev_src_start = ((c - 1) * num_src) // num_communities
                    prev_src_end = (c * num_src) // num_communities
                    reverse_src = torch.randint(prev_src_start, prev_src_end, (boundary_noise // 2,), device=device)
                    reverse_dst = torch.randint(dst_start, dst_end, (boundary_noise // 2,), device=device)
                    all_noise_src.append(reverse_src)
                    all_noise_dst.append(reverse_dst)

                # Remote bridging
                if module_switches[3] == 1 and c % 3 == 0 and c < num_communities - 2:
                    bridge_nodes = min(3, src_end - src_start)
                    bridge_src = torch.randint(src_start, src_end, (bridge_nodes,), device=device)

                    far_community = (c + num_communities // 2) % num_communities
                    far_dst_start = (far_community * num_dst) // num_communities
                    far_dst_end = ((far_community + 1) * num_dst) // num_communities

                    for bridge in bridge_src:
                        bridge_edges = edges_per_community_base // 10
                        far_dst = torch.randint(far_dst_start, far_dst_end, (bridge_edges,), device=device)
                        src_tensor = torch.full((bridge_edges,), bridge, device=device)
                        all_noise_src.append(src_tensor)
                        all_noise_dst.append(far_dst)

    # ========== Strategy 3: Feature Perturbation Induced Noise ==========
    if noise_per_type[2] > 0 and src_features is not None and dst_features is not None:
        num_patterns = 5  # Create 5 different pseudo-similarity patterns
        edges_per_pattern = noise_per_type[2] // num_patterns

        for p in range(num_patterns):
            # Select a different subset of feature dimensions for each pattern
            feature_dim = src_features.shape[1]

            # Select 20-40% of dimensions as "similar" dimensions
            num_similar_dims = int(feature_dim * (0.2 + 0.2 * p / num_patterns))
            similar_dims = torch.randperm(feature_dim, device=device)[:num_similar_dims]
            dissimilar_dims = torch.randperm(feature_dim, device=device)[num_similar_dims:]

            # Normalize features for similarity calculation
            src_feat_norm = src_features / (torch.norm(src_features, dim=1, keepdim=True) + 1e-8)
            dst_feat_norm = dst_features / (torch.norm(dst_features, dim=1, keepdim=True) + 1e-8)

            # Compute similarity on partial dimensions
            src_feat_partial = src_feat_norm[:, similar_dims]
            dst_feat_partial = dst_feat_norm[:, similar_dims]

            # Compute similarity matrix (use mini-batch to save memory)
            batch_size = min(100, num_src)
            top_k = min(edges_per_pattern // batch_size + 1, num_dst)

            for i in range(0, min(batch_size, num_src), 1):
                src_batch = src_feat_partial[i:i + 1]

                # Compute similarity with all targets
                similarities = torch.mm(src_batch, dst_feat_partial.t()).squeeze(0)

                # Add noise to create pseudo-similarity
                noise_factor = 0.3 + 0.1 * p  # Different noise intensity for different patterns
                similarities += torch.randn_like(similarities) * noise_factor

                # Select top-k but exclude the most similar ones (avoid real similar nodes)
                values, indices = torch.topk(similarities, min(top_k + 5, num_dst))

                # Skip top 5 most similar, select 6th to top_k+5
                selected_indices = indices[5:5 + top_k]

                if len(selected_indices) > 0:
                    src_tensor = torch.full((len(selected_indices),), i, device=device)
                    all_noise_src.append(src_tensor)
                    all_noise_dst.append(selected_indices)

    # ========== Strategy 4: Multi-modal Mixed Noise ==========
    if noise_per_type[3] > 0:
        # Distribute noise edges to 4 different modes
        mode_edges = noise_per_type[3] // 4

        # Mode 1: Local dense clique (requires local attention)
        num_cliques = 5
        clique_size = min(10, min(num_src, num_dst) // 10)
        edges_per_clique = mode_edges // num_cliques

        for _ in range(num_cliques):
            # Randomly select clique center
            src_center = torch.randint(0, num_src, (1,), device=device).item()
            dst_center = torch.randint(0, num_dst, (1,), device=device).item()

            # Create local dense connection
            src_clique = torch.randint(max(0, src_center - clique_size // 2),
                                       min(num_src, src_center + clique_size // 2),
                                       (edges_per_clique,), device=device)
            dst_clique = torch.randint(max(0, dst_center - clique_size // 2),
                                       min(num_dst, dst_center + clique_size // 2),
                                       (edges_per_clique,), device=device)

            all_noise_src.append(src_clique)
            all_noise_dst.append(dst_clique)

        # Mode 2: Long-range sparse bridge (requires global attention)
        num_bridges = 20
        edges_per_bridge = mode_edges // num_bridges

        for _ in range(num_bridges):
            # Select node pairs far apart
            src_bridge = torch.randint(0, num_src // 3, (edges_per_bridge,), device=device)
            dst_bridge = torch.randint(2 * num_dst // 3, num_dst, (edges_per_bridge,), device=device)

            all_noise_src.append(src_bridge)
            all_noise_dst.append(dst_bridge)

        # Mode 3: Periodic connection (requires specific frequency attention)
        period = max(5, min(num_src, num_dst) // 20)
        num_periods = mode_edges // period

        for i in range(num_periods):
            # Create periodic connection: connect every 'period' nodes
            src_periodic = (i * period) % num_src
            dst_periodic = torch.arange(i % period, min(num_dst, i % period + period),
                                        period, device=device)

            if len(dst_periodic) > 0:
                src_tensor = torch.full((len(dst_periodic),), src_periodic, device=device)
                all_noise_src.append(src_tensor)
                all_noise_dst.append(dst_periodic)

        # Mode 4: Asymmetric connection (Pattern where A connects to B but B doesn't connect to A)
        asym_edges = mode_edges

        # Create unidirectional connection chain
        for _ in range(10):  # 10 chains
            chain_length = asym_edges // 10

            # Create chain connection: src[i] -> dst[i] -> src[i+1] -> dst[i+1] ...
            src_chain = torch.randint(0, num_src, (chain_length,), device=device)
            dst_chain = torch.randint(0, num_dst, (chain_length,), device=device)

            # Ensure asymmetry: add only forward edges
            all_noise_src.append(src_chain)
            all_noise_dst.append(dst_chain)

    # Combine all noise edges
    if all_noise_src:
        noise_src = torch.cat(all_noise_src)
        noise_dst = torch.cat(all_noise_dst)

        # Fast deduplication
        edge_ids = noise_src * num_dst + noise_dst
        unique_edge_ids = torch.unique(edge_ids)

        # Restore source and target nodes
        noise_src_unique = unique_edge_ids // num_dst
        noise_dst_unique = unique_edge_ids % num_dst

        # Filter out existing edges
        if len(existing_edges) > 0:
            mask = torch.ones(len(noise_src_unique), dtype=torch.bool, device=device)

            if len(existing_edges) < 10000:
                src_orig_tensor = torch.tensor(list(e[0] for e in existing_edges), device=device)
                dst_orig_tensor = torch.tensor(list(e[1] for e in existing_edges), device=device)

                for i in range(len(src_orig_tensor)):
                    mask &= ~((noise_src_unique == src_orig_tensor[i]) &
                              (noise_dst_unique == dst_orig_tensor[i]))
            else:
                noise_pairs = set(zip(noise_src_unique.cpu().numpy(),
                                      noise_dst_unique.cpu().numpy()))
                new_edges = list(noise_pairs - existing_edges)

                if new_edges:
                    noise_src_unique = torch.tensor([e[0] for e in new_edges], device=device)
                    noise_dst_unique = torch.tensor([e[1] for e in new_edges], device=device)
                else:
                    noise_src_unique = torch.tensor([], device=device)
                    noise_dst_unique = torch.tensor([], device=device)

                mask = None

            if mask is not None:
                noise_src_unique = noise_src_unique[mask]
                noise_dst_unique = noise_dst_unique[mask]

        # Add noise edges to graph
        if len(noise_src_unique) > 0:
            hetero_graph.add_edges(noise_src_unique, noise_dst_unique, etype=noise_etype)

            # Add reverse edges
            if noise_etype == 'it':
                hetero_graph.add_edges(noise_dst_unique, noise_src_unique, etype='ti')
            elif noise_etype == 'hi':
                hetero_graph.add_edges(noise_dst_unique, noise_src_unique, etype='ih')

            if verbose:
                print(f"Successfully added {len(noise_src_unique)} noise edges")
        else:
            if verbose:
                print("All generated noise edges duplicate existing edges, none added")
    else:
        if verbose:
            print("No noise edges generated")

    return hetero_graph


