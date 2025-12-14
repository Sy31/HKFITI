import os
import dgl
import numpy as np
import model
from model import Model
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score
from sklearn.model_selection import KFold
from dgl.dataloading.negative_sampler import GlobalUniform
from tqdm import tqdm
import time
import torch
from utils import compute_loss, cos_sim
from utils import set_seed
from utils import remove_unseen_nodes, negative_sampling, add_redundant_noise
set_seed(100)
# ============ Calibration Evaluation Functions ============

def get_model_filename(dataset, has_regularization=False, fold1=False, skip_cv=False):
    parts = ['model', dataset]

    if has_regularization:
        parts.append('reg')

    if fold1:
        parts.append('fold1')
    elif skip_cv and dataset == 'bindingdb':
        parts.append('direct')

    return '_'.join(parts) + '.pt'


def predict_scores(model, node_embeddings, src_indices, dst_indices, device='cpu'):
    """Universal score prediction function, compatible with all predictor types"""
    import torch
    import numpy as np

    src_embeddings = node_embeddings['ingredient'][src_indices]
    dst_embeddings = node_embeddings['target'][dst_indices]

    if isinstance(src_embeddings, np.ndarray):
        src_embeddings = torch.from_numpy(src_embeddings).float()
        dst_embeddings = torch.from_numpy(dst_embeddings).float()

    # Process according to predictor type
    with torch.no_grad():
        from model import MLPScorePredictor, MixedPredictor, ScorePredictor

        if isinstance(model.pred, (MLPScorePredictor, MixedPredictor)):
            # For MLPScorePredictor and MixedPredictor, concatenate embeddings
            edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1).to(device)

            if isinstance(model.pred, MLPScorePredictor):
                # MLPScorePredictor: Use mlp directly
                scores = model.pred.mlp(edge_embeddings).squeeze(-1)
            else:  # MixedPredictor
                # For MixedPredictor, use full fusion prediction during validation
                # Calculate dot product scores
                dot_scores = (src_embeddings.to(device) * dst_embeddings.to(device)).sum(dim=1)
                # Calculate MLP scores
                mlp_scores = model.pred.mlp_predictor.mlp(edge_embeddings).squeeze(-1)

                # Combine scores based on fusion method
                if model.pred.fusion_method == 'residual':
                    scores = dot_scores + model.pred.residual_scale * mlp_scores
                elif model.pred.fusion_method == 'weighted':
                    if model.pred.alpha is not None:
                        alpha = torch.sigmoid(model.pred.alpha)
                    else:
                        alpha = 0.5
                    scores = alpha * dot_scores + (1 - alpha) * mlp_scores
                elif model.pred.fusion_method == 'gated':
                    # Gated fusion requires calculating gate weights
                    gates = model.pred.gate_net(edge_embeddings)
                    scores = gates[:, 0] * dot_scores + gates[:, 1] * mlp_scores
                else:
                    # Default to residual
                    scores = dot_scores + mlp_scores

        elif isinstance(model.pred, ScorePredictor):
            # ScorePredictor: Dot product
            scores = (src_embeddings.to(device) * dst_embeddings.to(device)).sum(dim=1)

        else:
            raise ValueError(f"Unknown predictor type: {type(model.pred)}")

    return scores.cpu().numpy()


def find_optimal_threshold(scores, labels):
    """Find optimal F1 threshold via precision-recall curve"""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    return optimal_threshold, optimal_f1


def create_train_graph_without_val_edges(hetero_graph, train_edges_idx, val_edges_idx, device):
    """
    Create training graph excluding validation edges
    Applicable to all datasets (herb, DrugBank, Davis, BindingDB)
    """
    # Get all 'it' edges
    src_all, dst_all = hetero_graph.edges(etype='it')

    # Create training edge mask
    train_mask = torch.zeros(len(src_all), dtype=torch.bool)
    train_mask[train_edges_idx] = True

    # Keep only training edges
    train_src = src_all[train_mask]
    train_dst = dst_all[train_mask]

    # Construct new graph data
    graph_data = {
        ('ingredient', 'it', 'target'): (train_src, train_dst),
        ('target', 'ti', 'ingredient'): (train_dst, train_src)
    }

    # Add other edge types (similarity edges, herb-ingredient edges, etc.)
    for etype in hetero_graph.canonical_etypes:
        src_type, edge_type, dst_type = etype
        if edge_type not in ['it', 'ti']:
            # Keep all other edge types
            src, dst = hetero_graph.edges(etype=etype)
            graph_data[etype] = (src, dst)

    # Get node counts from original graph to ensure new graph includes all nodes
    num_nodes_dict = {ntype: hetero_graph.num_nodes(ntype) for ntype in hetero_graph.ntypes}

    # Create new graph, explicitly specifying node counts
    train_graph = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

    # Copy node features
    for ntype in hetero_graph.ntypes:
        if 'features' in hetero_graph.nodes[ntype].data:
            train_graph.nodes[ntype].data['features'] = hetero_graph.nodes[ntype].data['features']

    return train_graph.to(device)


def generate_true_negative_samples(positive_pairs_set, num_samples, num_src, num_dst, max_attempts=100):
    """
    Generate true negative samples (ensuring they are not in the positive sample set)
    """
    neg_src = []
    neg_dst = []

    for i in range(num_samples):
        found = False
        for _ in range(max_attempts):
            s = np.random.randint(0, num_src)
            d = np.random.randint(0, num_dst)
            if (s, d) not in positive_pairs_set:
                neg_src.append(s)
                neg_dst.append(d)
                found = True
                break

        if not found:
            # If not found, choose randomly (rarely happens)
            neg_src.append(np.random.randint(0, num_src))
            neg_dst.append(np.random.randint(0, num_dst))

    return np.array(neg_src), np.array(neg_dst)


def train(args, hetero_graph, rel_list, device, predictor_type="md", pre_training_embeddings=None):
    """
    Training function preventing data leakage - Noise added only to training graph
    Applicable to all datasets (herb, DrugBank, Davis, BindingDB)
    """
    print("Using pretrained embeddings!")

    predictor_info = {
        'mlp': 'MLP Predictor',
        'dot': 'Dot Product Predictor',
        'md': 'Mixed Predictor (Dot + MLP Residual)'
    }
    print(f"\nUsing {predictor_info.get(predictor_type, predictor_type)} for link prediction")
    print(f"Dataset: {args.dataset}")

    # Get main prediction edges (do not add noise to original graph)
    it_eids = hetero_graph.edges(etype='it', form='eid')

    # Collect all positive pairs for negative sampling (positives from original graph)
    src_all, dst_all = hetero_graph.edges(etype='it')
    all_positive_pairs = set(zip(src_all.cpu().numpy(), dst_all.cpu().numpy()))
    num_ingredients = hetero_graph.num_nodes('ingredient')
    num_targets = hetero_graph.num_nodes('target')

    # K-Fold Cross Validation
    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=411)
    fold = 0
    results = []

    # Explain noise configuration once before K-fold starts
    if args.edge_noise_ratio != 0:
        print(f"\nEdge Noise Configuration:")
        print(f"  Ratio: {args.edge_noise_ratio}")
        print(f"  Distribution: Star={args.edge_noise_dis[0]:.2f}, Repeat={args.edge_noise_dis[1]:.2f}, "
              f"Hierarchy={args.edge_noise_dis[2]:.2f}, Symmetric(Enhance)={args.edge_noise_dis[3]:.2f}, Cluster(Enhance)={args.edge_noise_dis[4]:.2f}")
        print(f"  Type: {args.edge_noise_etype}")
        print(f"  Note: Noise is added independently to each fold's training graph; validation set remains clean.")

    if args.feature_noise_std != 0:
        from utils import add_gaussian_noise_to_features
        print(f"\nFeature Noise Configuration:")
        print(f"  Std Dev: {args.feature_noise_std}")
        print(f"  Node Types: {args.feature_noise_types}")
        print(f"  Note: Noise added independently before each fold.")

    # Check if regularization is used
    has_regularization = (getattr(args, 'lambda_head_div', 0) != 0 or
                          getattr(args, 'lambda_head_entropy', 0) != 0)

    for train_idx, val_idx in kf.split(it_eids, it_eids):
        print(f"\nStart Training Fold-{fold + 1}/{args.k_fold}")
        start_time = time.time()

        # Create training graph without validation edges
        train_graph = create_train_graph_without_val_edges(hetero_graph, train_idx, val_idx, device)
        original_edges = train_graph.num_edges('it')

        # Apply feature noise (independent for each fold)
        if args.feature_noise_std != 0:
            from utils import add_gaussian_noise_to_features
            noisy_initial_features = add_gaussian_noise_to_features(
                {'pretrained': hetero_graph.ndata['features']},
                args.feature_noise_std,
                args.feature_noise_types,
                device
            )
            # Update features of training graph
            train_graph.ndata['features'] = noisy_initial_features['pretrained']

        # Add edge noise only to training graph, not affecting validation
        if args.edge_noise_ratio != 0:
            # Add noise silently
            train_graph = add_redundant_noise(train_graph, args.edge_noise_ratio, args.edge_noise_dis,
                                              noise_etype=args.edge_noise_etype, device=device, verbose=False)
            noise_edges = train_graph.num_edges('it') - original_edges
            print(f"  Train Graph: {original_edges} edges + {noise_edges} noise = {train_graph.num_edges('it')} edges")
        else:
            print(f"  Train Graph: {train_graph.num_edges('it')} 'it' edges")

        print(f"  Validation Set: {len(val_idx)} edges")

        # Prepare training dataloader (using noisy training graph)
        train_eid_dict = {}
        for etype in train_graph.canonical_etypes:
            train_eid_dict[etype] = train_graph.edges(etype=etype, form='eid')

        train_negative_sampler = GlobalUniform(args.k)
        train_sampler = dgl.dataloading.NeighborSampler([args.k] * args.num_layers)
        train_sampler = dgl.dataloading.as_edge_prediction_sampler(train_sampler,
                                                                   negative_sampler=train_negative_sampler)
        train_dataloader = dgl.dataloading.DataLoader(
            train_graph,  # Use noisy training graph
            train_eid_dict,
            train_sampler,
            device=device,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0
        )

        # Prepare validation data (using original graph edges, no noise)
        val_src = src_all[val_idx].cpu().numpy()
        val_dst = dst_all[val_idx].cpu().numpy()

        # Generate true negative samples for validation (based on original graph)
        print(f"  Generating validation negative samples...")
        val_neg_src, val_neg_dst = generate_true_negative_samples(
            all_positive_pairs, len(val_src), num_ingredients, num_targets
        )

        # Create and train model
        model = Model(args, rel_list, predictor_type=predictor_type).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        lr_sche = torch.optim.lr_scheduler.StepLR(opt, args.lr_period, args.lr_decay)

        loss_values = []
        for epoch in tqdm(range(args.num_epochs), desc=f"Fold-{fold + 1}"):
            model.train()
            epoch_loss = []
            for step, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(train_dataloader):
                input_features = blocks[0].srcdata['features']
                pos_score, neg_score, reg_terms = model(args, positive_graph, negative_graph, blocks, input_features)
                loss = compute_loss(pos_score, neg_score, rel_list[0])

                # Add regularization
                lambda_div = getattr(args, 'lambda_head_div', 0.0)
                lambda_ent = getattr(args, 'lambda_head_entropy', 0.0)
                if lambda_div != 0 or lambda_ent != 0:
                    loss = loss + lambda_div * reg_terms['div'] + lambda_ent * reg_terms['ent']

                epoch_loss.append(loss.item())
                opt.zero_grad()
                loss.backward()
                opt.step()
            lr_sche.step()
            loss_values.append(np.array(epoch_loss).mean())

        end_time = time.time()
        training_time = end_time - start_time

        # Evaluate using original graph (no noise) for inference
        model.eval()
        s_time = time.time()

        with torch.no_grad():
            # Use original graph features (no noise) for forward pass
            all_features = hetero_graph.ndata['features']

            # Feature mapping (always use pretrained embeddings)
            mapped_features = {}
            if 'herb' in all_features and hasattr(model, 'mapping_herb'):
                mapped_features['herb'] = model.mapping_herb(all_features['herb'])
            if 'ingredient' in all_features:
                mapped_features['ingredient'] = model.mapping_ingredient(all_features['ingredient'])
            if 'target' in all_features:
                mapped_features['target'] = model.mapping_target(all_features['target'])

            # Create blocks for inference (using original graph)
            blocks = []
            for _ in range(args.num_layers):
                block = dgl.to_block(hetero_graph,
                                     dst_nodes={ntype: torch.arange(hetero_graph.num_nodes(ntype), device=device)
                                                for ntype in hetero_graph.ntypes})
                blocks.append(block)

            blocks[0].srcdata['features'] = mapped_features
            node_embeddings = model.HeteroGNN(blocks, mapped_features)

            # Evaluate validation set (based on original graph embeddings)
            if predictor_type in ['mlp', 'md']:
                positive_res = predict_scores(model, node_embeddings, val_src, val_dst, device)
                negative_res = predict_scores(model, node_embeddings, val_neg_src, val_neg_dst, device)
            else:
                node_embeddings = {k: v.to('cpu') for k, v in node_embeddings.items()}
                positive_res = cos_sim(np.array(node_embeddings['ingredient'][val_src]),
                                       np.array(node_embeddings['target'][val_dst]))
                negative_res = cos_sim(np.array(node_embeddings['ingredient'][val_neg_src]),
                                       np.array(node_embeddings['target'][val_neg_dst]))

        e_time = time.time()
        inference_time = e_time - s_time

        # Calculate evaluation metrics
        positive_labels = np.ones(len(positive_res))
        negative_labels = np.zeros(len(negative_res))

        all_scores = np.concatenate([positive_res, negative_res], axis=0)
        all_labels = np.concatenate([positive_labels, negative_labels], axis=0)

        auroc = roc_auc_score(all_labels, all_scores)
        auprc = average_precision_score(all_labels, all_scores)

        optimal_threshold, optimal_f1 = find_optimal_threshold(all_scores, all_labels)
        predictions = (all_scores >= optimal_threshold).astype(int)
        f1 = f1_score(all_labels, predictions)

        # ============ Calibration Evaluation ============
        print(
            f"Fold-{fold + 1} - AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}, F1: {f1:.3f} (threshold: {optimal_threshold:.3f}), "
            f"Training_time: {training_time:.3f}, Inference_time: {inference_time:.3f}")

        # Save first fold model (if needed)
        if fold == 0 and hasattr(args, 'save_fold1_model') and args.save_fold1_model:
            os.makedirs('saved_models', exist_ok=True)
            fold1_filename = get_model_filename(args.dataset, has_regularization, fold1=True)
            fold1_save_path = os.path.join('saved_models', fold1_filename)

            save_dict = {
                'model_state_dict': model.state_dict(),
                'args': args,
                'rel_list': rel_list,
                'fold': fold + 1,
                'auroc': auroc,
                'auprc': auprc,
                'f1': f1,
                'node_embeddings': {k: v.cpu() for k, v in node_embeddings.items()},
                'hetero_graph_info': {
                    'num_nodes': {ntype: hetero_graph.num_nodes(ntype) for ntype in hetero_graph.ntypes},
                    'num_edges': {etype: hetero_graph.num_edges(etype) for etype in hetero_graph.etypes}
                }
            }
            torch.save(save_dict, fold1_save_path)
            print(f"  ✓ Fold 1 model saved to: {fold1_save_path}")

        fold += 1

        # Modify result logging, use dict format to store more info
        fold_result = {
            'auroc': auroc,
            'auprc': auprc,
            'f1': f1,
            'training_time': training_time
        }

        results.append(fold_result)

    # Calculate average performance - Compatible with new and old formats
    if results and isinstance(results[0], dict):
        # New format
        mean_auroc = np.mean([r['auroc'] for r in results])
        mean_auprc = np.mean([r['auprc'] for r in results])
        mean_f1 = np.mean([r['f1'] for r in results])
        mean_time = np.mean([r['training_time'] for r in results])

        std_auroc = np.std([r['auroc'] for r in results])
        std_auprc = np.std([r['auprc'] for r in results])
        std_f1 = np.std([r['f1'] for r in results])
        std_time = np.std([r['training_time'] for r in results])

        print(f"\n{fold}-fold average performance:\n"
              f"AUROC: {mean_auroc:.3f} ± {std_auroc:.3f}\n"
              f"AUPRC: {mean_auprc:.3f} ± {std_auprc:.3f}\n"
              f"F1: {mean_f1:.3f} ± {std_f1:.3f}\n"
              f"Time: {mean_time:.3f} ± {std_time:.3f}")

        # Summary of calibration metrics
    else:
        # Compatible with old format (list of tuples)
        results_arr = np.array(results)
        mean_auroc = np.mean(results_arr[:, 0])
        mean_auprc = np.mean(results_arr[:, 1])
        mean_f1 = np.mean(results_arr[:, 2])
        mean_time = np.mean(results_arr[:, 3])

        std_auroc = np.std(results_arr[:, 0])
        std_auprc = np.std(results_arr[:, 1])
        std_f1 = np.std(results_arr[:, 2])
        std_time = np.std(results_arr[:, 3])

        print(f"\n{fold}-fold average performance:\n"
              f"AUROC: {mean_auroc:.3f} ± {std_auroc:.3f}\n"
              f"AUPRC: {mean_auprc:.3f} ± {std_auprc:.3f}\n"
              f"F1: {mean_f1:.3f} ± {std_f1:.3f}\n"
              f"Time: {mean_time:.3f} ± {std_time:.3f}")

    # Save final model (if needed) - No noise added
    if args.save_model:
        print("\nTraining final model for saving...")

        final_model = Model(args, rel_list, predictor_type=predictor_type).to(device)
        opt = torch.optim.Adam(final_model.parameters(), lr=args.lr, weight_decay=args.wd)
        lr_sche = torch.optim.lr_scheduler.StepLR(opt, args.lr_period, args.lr_decay)

        # Prepare full data DataLoader (no noise)
        all_eid_dict = {etype: hetero_graph.edges(etype=etype, form='eid')
                        for etype in hetero_graph.etypes}

        negative_sampler = GlobalUniform(args.k)
        sampler = dgl.dataloading.NeighborSampler([args.k] * args.num_layers)
        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, negative_sampler=negative_sampler)

        dataloader = dgl.dataloading.DataLoader(
            hetero_graph,
            all_eid_dict,
            sampler,
            device=device,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0
        )

        # Train final model
        for epoch in tqdm(range(args.num_epochs), desc="Training final model"):
            final_model.train()
            for step, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(dataloader):
                input_features = blocks[0].srcdata['features']
                pos_score, neg_score, reg_terms = final_model(args, positive_graph, negative_graph, blocks,
                                                              input_features)
                loss = compute_loss(pos_score, neg_score, rel_list[0])

                lambda_div = getattr(args, 'lambda_head_div', 0.0)
                lambda_ent = getattr(args, 'lambda_head_entropy', 0.0)
                if lambda_div != 0 or lambda_ent != 0:
                    loss = loss + lambda_div * reg_terms['div'] + lambda_ent * reg_terms['ent']

                opt.zero_grad()
                loss.backward()
                opt.step()
            lr_sche.step()

        # Save model and info
        save_dict = {
            'model_state_dict': final_model.state_dict(),
            'args': args,
            'rel_list': rel_list,
            'predictor_type': predictor_type,
            'dataset': args.dataset,
            'mean_auroc': mean_auroc,
            'mean_auprc': mean_auprc,
            'mean_f1': mean_f1,
            'std_auroc': std_auroc,
            'std_auprc': std_auprc,
            'std_f1': std_f1
        }

        os.makedirs('saved_models', exist_ok=True)
        model_filename = get_model_filename(args.dataset, has_regularization)
        model_path = os.path.join('saved_models', model_filename)
        torch.save(save_dict, model_path)
        print(f"Model saved to: {model_path}")
        print(
            f"Model Performance - AUROC: {mean_auroc:.3f} ± {std_auroc:.3f}, AUPRC: {mean_auprc:.3f} ± {std_auprc:.3f}, F1: {mean_f1:.3f} ± {std_f1:.3f}")

    # Mode 1: Ligand Profile Analysis (visualize_embeddings.py)
    if args.visualize_profile == 1:
        from visualize_embeddings import run_ligand_profile_analysis
        print("\nRunning post-training Ligand Profile Analysis (Mode 1)...")

        # Extract initial features from hetero_graph
        initial_features_dict = {
            'pretrained': hetero_graph.ndata['features']
        }

        # Check if node_embeddings exists (exists when save_fold1_model=True)
        if 'node_embeddings' not in locals() and 'node_embeddings' not in globals():
            print("Warning: 'node_embeddings' variable not found.")
            print("Ligand Profile Analysis (Mode 1) may not be able to access trained embeddings.")
            print("Please ensure --save_fold1_model is set.")
            trained_embeds = None
        else:
            trained_embeds = node_embeddings  # Use node_embeddings saved in KFold loop

        run_ligand_profile_analysis(args, hetero_graph, initial_features_dict,
                                    model=model, trained_embeddings=trained_embeds,
                                    filter_all_isolated=args.filter_all_isolated)

    # Mode 2: Label Visualization (visualize_embeddings2.py)
    elif args.visualize_profile == 2:
        from visualize_embeddings2 import visualize_after_training
        print("\nRunning post-training Label Visualization (Mode 2)...")

        # Prepare initial features (same as Mode 1)
        initial_features_dict = {
            'pretrained': hetero_graph.ndata['features']
        }

        # Call modified function in visualize_embeddings2.py
        # It handles label loading, model inference, and output path definition
        visualize_after_training(model, hetero_graph, initial_features_dict, args)

    # Mode 3: Simple Ingredient-Target Visualization
    elif args.visualize_profile == 3:
        from visualize_embeddings2 import run_simple_ingredient_target_visualization
        print("\nRunning post-training Ingredient-Target Visualization (Mode 3)...")

        # Prepare initial features
        initial_features_dict = {
            'pretrained': hetero_graph.ndata['features']
        }

        # Call simple visualization function, passing model and saved pre-training embeddings
        run_simple_ingredient_target_visualization(args, hetero_graph, initial_features_dict,
                                                   model=model,
                                                   pre_training_embeddings=pre_training_embeddings)

    elif args.visualize_profile == 4:
        from visualize_embeddings2 import run_tsne_ingredient_target_visualization
        print("\nRunning post-training Ingredient-Target t-SNE Visualization (Mode 4)...")

        # Prepare initial features
        initial_features_dict = {
            'pretrained': hetero_graph.ndata['features']
        }

        # Call t-SNE visualization function, passing model and saved pre-training embeddings
        run_tsne_ingredient_target_visualization(args, hetero_graph, initial_features_dict,
                                                 model=model,
                                                 pre_training_embeddings=pre_training_embeddings)

    return mean_auroc, mean_auprc, mean_f1


def unseen_train(args, hetero_graph, rel_list, device, predictor_type="md", pre_training_embeddings=None):
    """
    Cold Start Training (Improved) - Noise added only to training graph
    Applicable to unseen setting for all datasets
    """
    # Determine unseen node type
    if args.unseen_setting == 1:
        # unseen ingredients/drugs
        node_type = 'ingredient'
        total_nodes = hetero_graph.num_nodes('ingredient')
        unseen_nodes = np.arange(total_nodes)
        print(f"\nCold Start Setting: Unseen {node_type}s (Total {total_nodes})")
    elif args.unseen_setting == 2:
        # unseen targets
        node_type = 'target'
        total_nodes = hetero_graph.num_nodes('target')
        unseen_nodes = np.arange(total_nodes)
        print(f"\nCold Start Setting: Unseen {node_type}s (Total {total_nodes})")
    else:
        raise ValueError(f"Unsupported unseen_setting: {args.unseen_setting}")

    # Collect all positive pairs
    src_all, dst_all = hetero_graph.edges(etype='it')
    all_positive_pairs = set(zip(src_all.cpu().numpy(), dst_all.cpu().numpy()))

    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=411)
    folds = list(kf.split(unseen_nodes))
    results = []

    for fold in range(args.k_fold):
        print(f"\nStart Training Fold-{fold + 1}/{args.k_fold}")

        model = Model(args, rel_list, predictor_type=predictor_type).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        lr_sche = torch.optim.lr_scheduler.StepLR(opt, args.lr_period, args.lr_decay)

        train_indices, val_indices = folds[fold]
        val_nodes = torch.tensor(val_indices, device=device)

        # Get all 'it' edges from original graph
        src, dst = hetero_graph.edges(etype='it')

        # Find edges involving test nodes (these are true test positives)
        if node_type == 'ingredient':
            test_mask = torch.isin(src, val_nodes)
        else:
            test_mask = torch.isin(dst, val_nodes)

        # Test set positive edges
        test_pos_src = src[test_mask].cpu().numpy()
        test_pos_dst = dst[test_mask].cpu().numpy()

        # Build training graph (remove test nodes)
        train_graph, _, _ = remove_unseen_nodes(node_type, hetero_graph, val_indices)
        train_graph = train_graph.to(device)

        # Apply feature noise
        if args.feature_noise_std != 0:
            from utils import add_gaussian_noise_to_features
            noisy_initial_features = add_gaussian_noise_to_features(
                {'pretrained': train_graph.ndata['features']},
                args.feature_noise_std,
                args.feature_noise_types,
                device
            )
            train_graph.ndata['features'] = noisy_initial_features['pretrained']

        # Add edge noise only to training graph
        if args.edge_noise_ratio != 0:
            print(f"  Adding edge noise to Fold-{fold + 1} training graph...")
            train_graph = add_redundant_noise(train_graph, args.edge_noise_ratio, args.edge_noise_dis,
                                              noise_etype=args.edge_noise_etype, device=device)

        print(f"  Train Graph: {train_graph.num_nodes('ingredient')} ingredients, {train_graph.num_nodes('target')} targets")
        print(f"  Test Set: {len(test_pos_src)} edges")

        # Generate negative samples for test set
        if node_type == 'ingredient':
            # For each test ingredient, randomly select target as negative
            neg_edges_src = []
            neg_edges_dst = []

            remaining_targets = train_graph.nodes('target').cpu().numpy()

            for test_src in np.unique(test_pos_src):
                # Get true targets for this ingredient
                true_targets = test_pos_dst[test_pos_src == test_src]
                # Choose negative samples from remaining targets
                available_targets = np.setdiff1d(remaining_targets, true_targets)

                if len(available_targets) > 0:
                    # Generate one negative sample per positive sample
                    num_pos = len(true_targets)
                    neg_targets = np.random.choice(available_targets,
                                                   min(num_pos, len(available_targets)),
                                                   replace=False)
                    for neg_dst in neg_targets:
                        neg_edges_src.append(test_src)
                        neg_edges_dst.append(neg_dst)
        else:
            # For each test target, randomly select ingredient as negative
            neg_edges_src = []
            neg_edges_dst = []

            remaining_ingredients = train_graph.nodes('ingredient').cpu().numpy()

            for test_dst in np.unique(test_pos_dst):
                # Get true ingredients for this target
                true_ingredients = test_pos_src[test_pos_dst == test_dst]
                # Choose negative samples from remaining ingredients
                available_ingredients = np.setdiff1d(remaining_ingredients, true_ingredients)

                if len(available_ingredients) > 0:
                    # Generate one negative sample per positive sample
                    num_pos = len(true_ingredients)
                    neg_ingredients = np.random.choice(available_ingredients,
                                                       min(num_pos, len(available_ingredients)),
                                                       replace=False)
                    for neg_src in neg_ingredients:
                        neg_edges_src.append(neg_src)
                        neg_edges_dst.append(test_dst)

        # Train model
        eid_dict = {
            etype: train_graph.edges(etype=etype, form='eid')
            for etype in train_graph.etypes
        }

        negative_sampler = GlobalUniform(args.k)
        sampler = dgl.dataloading.NeighborSampler([args.k] * args.num_layers)
        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, negative_sampler=negative_sampler)

        dataloader = dgl.dataloading.DataLoader(
            train_graph,
            eid_dict,
            sampler,
            device=device,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0
        )

        # Training loop
        for epoch in tqdm(range(args.num_epochs), desc=f"Fold-{fold + 1}"):
            model.train()
            for step, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(dataloader):
                input_features = blocks[0].srcdata['features']
                pos_score, neg_score, reg_terms = model(args, positive_graph, negative_graph, blocks, input_features)
                loss = compute_loss(pos_score, neg_score, rel_list[0])

                lambda_div = getattr(args, 'lambda_head_div', 0.0)
                lambda_ent = getattr(args, 'lambda_head_entropy', 0.0)
                if lambda_div != 0 or lambda_ent != 0:
                    loss = loss + lambda_div * reg_terms['div'] + lambda_ent * reg_terms['ent']

                opt.zero_grad()
                loss.backward()
                opt.step()
            lr_sche.step()

        # Evaluate (using original graph, no noise)
        with torch.no_grad():
            # Use all node features from original graph for evaluation
            all_features = hetero_graph.ndata['features']

            # Feature mapping (always use pretrained embeddings)
            mapped_features = {}
            if 'herb' in all_features and hasattr(model, 'mapping_herb'):
                mapped_features['herb'] = model.mapping_herb(all_features['herb'])
            if 'ingredient' in all_features:
                mapped_features['ingredient'] = model.mapping_ingredient(all_features['ingredient'])
            if 'target' in all_features:
                mapped_features['target'] = model.mapping_target(all_features['target'])

            # Create blocks for inference
            blocks = []
            for _ in range(args.num_layers):
                block = dgl.to_block(hetero_graph,
                                     dst_nodes={ntype: torch.arange(hetero_graph.num_nodes(ntype), device=device)
                                                for ntype in hetero_graph.ntypes})
                blocks.append(block)

            blocks[0].srcdata['features'] = mapped_features
            node_embeddings = model.HeteroGNN(blocks, mapped_features)

            # Calculate test set scores
            if predictor_type in ['mlp', 'md']:
                positive_res = predict_scores(model, node_embeddings, test_pos_src, test_pos_dst, device)
                negative_res = predict_scores(model, node_embeddings, neg_edges_src, neg_edges_dst, device)
            else:
                node_embeddings = {k: v.to('cpu') for k, v in node_embeddings.items()}
                positive_res = cos_sim(np.array(node_embeddings['ingredient'][test_pos_src]),
                                       np.array(node_embeddings['target'][test_pos_dst]))
                negative_res = cos_sim(np.array(node_embeddings['ingredient'][neg_edges_src]),
                                       np.array(node_embeddings['target'][neg_edges_dst]))

        # Calculate evaluation metrics
        positive_labels = np.ones(len(positive_res))
        negative_labels = np.zeros(len(negative_res))

        all_scores = np.concatenate([positive_res, negative_res], axis=0)
        all_labels = np.concatenate([positive_labels, negative_labels], axis=0)

        auroc = roc_auc_score(all_labels, all_scores)
        auprc = average_precision_score(all_labels, all_scores)

        optimal_threshold, optimal_f1 = find_optimal_threshold(all_scores, all_labels)
        predictions = (all_scores >= optimal_threshold).astype(int)
        f1 = f1_score(all_labels, predictions)

        print(
            f"Fold-{fold + 1} - AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}, F1: {f1:.3f} (threshold: {optimal_threshold:.3f})")

        results.append((auroc, auprc, f1))

    # Calculate average results
    results_arr = np.array(results)
    mean_auroc = np.mean(results_arr[:, 0])
    mean_auprc = np.mean(results_arr[:, 1])
    mean_f1 = np.mean(results_arr[:, 2])
    std_auroc = np.std(results_arr[:, 0])
    std_auprc = np.std(results_arr[:, 1])
    std_f1 = np.std(results_arr[:, 2])

    print(f"\n{args.k_fold}-fold average performance:")
    print(f"AUROC: {mean_auroc:.3f} ± {std_auroc:.3f}")
    print(f"AUPRC: {mean_auprc:.3f} ± {std_auprc:.3f}")
    print(f"F1: {mean_f1:.3f} ± {std_f1:.3f}")

    return mean_auroc, mean_auprc, mean_f1

    # Visualization after training (unseen_train mode)
    if args.visualize_profile == 1:
        from visualize_embeddings import run_ligand_profile_analysis
        print("\nRunning post-training Ligand Profile Analysis (Mode 1)...")
        initial_features_dict = {
            'pretrained': hetero_graph.ndata['features']
        }
        # In unseen mode, node_embeddings might exist in the last fold
        trained_embeds = node_embeddings if 'node_embeddings' in locals() else None
        run_ligand_profile_analysis(args, hetero_graph, initial_features_dict,
                                    model=model, trained_embeddings=trained_embeds,
                                    filter_all_isolated=args.filter_all_isolated)




def train_bindingdb_with_test(args, hetero_graph, rel_list, test_edges, device, predictor_type="md"):
    """
    BindingDB Special Training Pipeline (Fixed) - Noise added only during training
    Includes cross-validation and independent test set evaluation
    Note: hetero_graph contains all nodes (train + test), but only training edges
    """
    print("\n==================== BindingDB Three-Layer Evaluation Architecture ====================")
    print("Phase 1: K-Fold Cross Validation (Model Development)")
    print("Phase 2: Full Data Training (Final Model)")
    print("Phase 3: Independent Test Set Evaluation (Real Performance)")
    print("===========================================================\n")

    # Phase 1: K-Fold Cross Validation (Use fixed train function)
    print(f"\n[Phase 1] Running {args.k_fold}-Fold Cross Validation...")
    cv_auroc, cv_auprc, cv_f1 = train(args, hetero_graph, rel_list, device, predictor_type)

    # Check if regularization is used
    has_regularization = (getattr(args, 'lambda_head_div', 0) != 0 or
                          getattr(args, 'lambda_head_entropy', 0) != 0)

    # Phase 2: Train final model using all training data (No noise added to final model)
    print("\n[Phase 2] Training final model using all training data...")
    final_model = Model(args, rel_list, predictor_type=predictor_type).to(device)
    opt = torch.optim.Adam(final_model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_sche = torch.optim.lr_scheduler.StepLR(opt, args.lr_period, args.lr_decay)

    # Prepare training DataLoader (Use only training edges, no noise)
    all_eid_dict = {etype: hetero_graph.edges(etype=etype, form='eid')
                    for etype in hetero_graph.etypes}

    negative_sampler = GlobalUniform(args.k)
    sampler = dgl.dataloading.NeighborSampler([args.k] * args.num_layers)
    sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, negative_sampler=negative_sampler)

    dataloader = dgl.dataloading.DataLoader(
        hetero_graph,
        all_eid_dict,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )

    # Train final model
    for epoch in tqdm(range(args.num_epochs), desc="Training final model"):
        final_model.train()
        epoch_loss = []
        for step, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(dataloader):
            input_features = blocks[0].srcdata['features']
            pos_score, neg_score, reg_terms = final_model(args, positive_graph, negative_graph, blocks, input_features)
            loss = compute_loss(pos_score, neg_score, rel_list[0])

            lambda_div = getattr(args, 'lambda_head_div', 0.0)
            lambda_ent = getattr(args, 'lambda_head_entropy', 0.0)
            if lambda_div != 0 or lambda_ent != 0:
                loss = loss + lambda_div * reg_terms['div'] + lambda_ent * reg_terms['ent']

            epoch_loss.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        lr_sche.step()

    # Phase 3: Evaluate on independent test set
    print("\n[Phase 3] Evaluating final model on independent test set...")
    final_model.eval()

    # Prepare test data
    test_src = test_edges['source'].values
    test_dst = test_edges['target'].values

    print(f"  Test set contains {len(test_src)} edges")
    print(f"  Drug ID Range: {test_src.min()}-{test_src.max()}")
    print(f"  Target ID Range: {test_dst.min()}-{test_dst.max()}")

    # Collect all positive pairs (including training and test sets)
    train_src, train_dst = hetero_graph.edges(etype='it')
    train_positive_pairs = set(zip(train_src.cpu().numpy(), train_dst.cpu().numpy()))
    test_positive_pairs = set(zip(test_src, test_dst))
    all_positive_pairs = train_positive_pairs.union(test_positive_pairs)

    num_drugs = hetero_graph.num_nodes('ingredient')
    num_targets = hetero_graph.num_nodes('target')

    print(f"  Total Drug Nodes in Graph: {num_drugs}")
    print(f"  Total Target Nodes in Graph: {num_targets}")

    # Generate test set negative samples (ensuring not in any positive set)
    print("  Generating test set negative samples...")
    neg_src, neg_dst = generate_true_negative_samples(
        all_positive_pairs, len(test_src), num_drugs, num_targets
    )

    with torch.no_grad():
        # Get embeddings for all nodes (including test nodes)
        all_features = hetero_graph.ndata['features']

        # Feature mapping (always use pretrained embeddings)
        mapped_features = {}
        if 'ingredient' in all_features:
            mapped_features['ingredient'] = final_model.mapping_ingredient(all_features['ingredient'])
        if 'target' in all_features:
            mapped_features['target'] = final_model.mapping_target(all_features['target'])

        # Create inference blocks (use full graph)
        blocks = []
        for _ in range(args.num_layers):
            block = dgl.to_block(hetero_graph,
                                 dst_nodes={ntype: torch.arange(hetero_graph.num_nodes(ntype), device=device)
                                            for ntype in hetero_graph.ntypes})
            blocks.append(block)

        blocks[0].srcdata['features'] = mapped_features
        node_embeddings = final_model.HeteroGNN(blocks, mapped_features)

        # Verify test set nodes are in embeddings
        max_test_src = test_src.max()
        max_test_dst = test_dst.max()
        print(f"  Drug Embedding Shape: {node_embeddings['ingredient'].shape}")
        print(f"  Target Embedding Shape: {node_embeddings['target'].shape}")

        if max_test_src >= node_embeddings['ingredient'].shape[0]:
            print(f"  Warning: Drug ID out of range in test set: {max_test_src}")
        if max_test_dst >= node_embeddings['target'].shape[0]:
            print(f"  Warning: Target ID out of range in test set: {max_test_dst}")

        # Evaluate positives and negatives
        if predictor_type in ['mlp', 'md']:
            pos_scores = predict_scores(final_model, node_embeddings, test_src, test_dst, device)
            neg_scores = predict_scores(final_model, node_embeddings, neg_src, neg_dst, device)
        else:
            node_embeddings = {k: v.to('cpu') for k, v in node_embeddings.items()}
            pos_scores = cos_sim(node_embeddings['ingredient'][test_src].numpy(),
                                 node_embeddings['target'][test_dst].numpy())
            neg_scores = cos_sim(node_embeddings['ingredient'][neg_src].numpy(),
                                 node_embeddings['target'][neg_dst].numpy())

    # Calculate test set performance
    pos_labels = np.ones(len(pos_scores))
    neg_labels = np.zeros(len(neg_scores))

    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([pos_labels, neg_labels])

    test_auroc = roc_auc_score(all_labels, all_scores)
    test_auprc = average_precision_score(all_labels, all_scores)

    optimal_threshold, optimal_f1 = find_optimal_threshold(all_scores, all_labels)
    predictions = (all_scores >= optimal_threshold).astype(int)
    test_f1 = f1_score(all_labels, predictions)

    # Print final results
    print("\n" + "=" * 60)
    print("BindingDB Final Evaluation Results:")
    print("=" * 60)
    print(f"Cross Validation (Development):")
    print(f"  AUROC: {cv_auroc:.4f}, AUPRC: {cv_auprc:.4f}, F1: {cv_f1:.4f}")
    print(f"\nIndependent Test Set (Real Generalization):")
    print(f"  AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}, F1: {test_f1:.4f}")
    print(f"  Test Positives: {len(pos_scores)}, Negatives: {len(neg_scores)}")
    print("=" * 60)

    # Save model (if needed)
    if args.save_model:
        os.makedirs('saved_models', exist_ok=True)
        model_filename = get_model_filename(args.dataset, has_regularization)
        save_path = os.path.join('saved_models', model_filename)

        save_dict = {
            'model_state_dict': final_model.state_dict(),
            'args': args,
            'dataset': 'bindingdb',
            'cv_performance': {'auroc': cv_auroc, 'auprc': cv_auprc, 'f1': cv_f1},
            'test_performance': {'auroc': test_auroc, 'auprc': test_auprc, 'f1': test_f1},
            'test_set_size': len(test_src)
        }
        torch.save(save_dict, save_path)
        print(f"\nFinal model saved to: {save_path}")

    return test_auroc, test_auprc, test_f1

    # Visualization after training (BindingDB Test Mode)

    if args.visualize_profile == 1:
        from visualize_embeddings import run_ligand_profile_analysis
        print("\nRunning post-training Ligand Profile Analysis (Mode 1)...")
        initial_features_dict = {
            'pretrained': hetero_graph.ndata['features']
        }
        run_ligand_profile_analysis(args, hetero_graph, initial_features_dict, model=final_model,
                                    filter_all_isolated=args.filter_all_isolated)


def train_bindingdb_direct(args, hetero_graph, rel_list, test_edges, device, predictor_type="md"):
    """
    BindingDB Direct Training Mode - Skip CV, train final model directly and test

    Args:
        args: Parameter config
        hetero_graph: Heterogeneous graph (all nodes, training edges only)
        rel_list: Relation list
        test_edges: Test set edges
        device: Device
        predictor_type: Predictor type ('mlp', 'dot', 'md')
    """
    print("\n==================== BindingDB Fast Training Mode ====================")
    print("Skipping K-Fold Cross Validation")
    print("Training final model directly with all training data")
    print("Evaluating performance on independent test set")
    print("===========================================================\n")

    print("Using pretrained embeddings!")
    predictor_info = {
        'mlp': 'MLP Predictor',
        'dot': 'Dot Product Predictor',
        'md': 'Mixed Predictor (Dot + MLP Residual)'
    }
    print(f"Using {predictor_info.get(predictor_type, predictor_type)} for link prediction")

    # Check if regularization is used
    has_regularization = (getattr(args, 'lambda_head_div', 0) != 0 or
                          getattr(args, 'lambda_head_entropy', 0) != 0)

    # Step 1: Directly train final model
    print("\n[Step 1] Training model using all training data...")
    final_model = Model(args, rel_list, predictor_type=predictor_type).to(device)
    opt = torch.optim.Adam(final_model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_sche = torch.optim.lr_scheduler.StepLR(opt, args.lr_period, args.lr_decay)

    # Prepare training DataLoader (Use training edges, optionally add noise)
    train_graph = hetero_graph.clone()  # Clone graph to avoid modifying original

    # Apply feature noise
    if args.feature_noise_std != 0:
        from utils import add_gaussian_noise_to_features
        print(f"\nAdding feature noise to training graph (std={args.feature_noise_std})...")
        noisy_initial_features = add_gaussian_noise_to_features(
            {'pretrained': train_graph.ndata['features']},
            args.feature_noise_std,
            args.feature_noise_types,
            device
        )
        train_graph.ndata['features'] = noisy_initial_features['pretrained']

    # Add edge noise if needed
    if args.edge_noise_ratio != 0:
        print(f"\nAdding edge noise to training graph (ratio={args.edge_noise_ratio})...")
        train_graph = add_redundant_noise(train_graph, args.edge_noise_ratio, args.edge_noise_dis,
                                          noise_etype=args.edge_noise_etype, device=device, verbose=True)
        print(f"Training Graph: {train_graph.num_edges('it')} edges (including noise)")
    else:
        print(f"Training Graph: {train_graph.num_edges('it')} edges")

    all_eid_dict = {etype: train_graph.edges(etype=etype, form='eid')
                    for etype in train_graph.etypes}

    negative_sampler = GlobalUniform(args.k)
    sampler = dgl.dataloading.NeighborSampler([args.k] * args.num_layers)
    sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, negative_sampler=negative_sampler)

    dataloader = dgl.dataloading.DataLoader(
        train_graph,
        all_eid_dict,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )

    # Training loop
    print(f"Start Training ({args.num_epochs} epochs)...")
    start_time = time.time()

    for epoch in tqdm(range(args.num_epochs), desc="Training Progress"):
        final_model.train()
        epoch_loss = []
        for step, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(dataloader):
            input_features = blocks[0].srcdata['features']
            pos_score, neg_score, reg_terms = final_model(args, positive_graph, negative_graph, blocks, input_features)
            loss = compute_loss(pos_score, neg_score, rel_list[0])

            # Add regularization
            lambda_div = getattr(args, 'lambda_head_div', 0.0)
            lambda_ent = getattr(args, 'lambda_head_entropy', 0.0)
            if lambda_div != 0 or lambda_ent != 0:
                loss = loss + lambda_div * reg_terms['div'] + lambda_ent * reg_terms['ent']

            epoch_loss.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        lr_sche.step()

        # Print average loss every 5 epochs
        if (epoch + 1) % 5 == 0:
            avg_loss = np.mean(epoch_loss)
            print(f"  Epoch {epoch + 1}/{args.num_epochs} - Avg Loss: {avg_loss:.4f}")

    training_time = time.time() - start_time
    print(f"Training complete, time: {training_time:.2f}s")

    # Step 2: Evaluate on independent test set
    print("\n[Step 2] Evaluating model on independent test set...")
    final_model.eval()

    # Prepare test data
    test_src = test_edges['source'].values
    test_dst = test_edges['target'].values

    print(f"  Test set contains {len(test_src)} edges")
    print(f"  Drug ID Range: {test_src.min()}-{test_src.max()}")
    print(f"  Target ID Range: {test_dst.min()}-{test_dst.max()}")

    # Collect all positive pairs (including training and test sets)
    train_src, train_dst = hetero_graph.edges(etype='it')
    train_positive_pairs = set(zip(train_src.cpu().numpy(), train_dst.cpu().numpy()))
    test_positive_pairs = set(zip(test_src, test_dst))
    all_positive_pairs = train_positive_pairs.union(test_positive_pairs)

    num_drugs = hetero_graph.num_nodes('ingredient')
    num_targets = hetero_graph.num_nodes('target')

    # Generate test set negative samples (ensuring not in any positive set)
    print("  Generating test set negative samples...")
    neg_src, neg_dst = generate_true_negative_samples(
        all_positive_pairs, len(test_src), num_drugs, num_targets
    )

    # Perform inference (using original graph, no noise)
    with torch.no_grad():
        # Get all node embeddings
        all_features = hetero_graph.ndata['features']

        # Feature mapping (always use pretrained embeddings)
        mapped_features = {}
        if 'ingredient' in all_features:
            mapped_features['ingredient'] = final_model.mapping_ingredient(all_features['ingredient'])
        if 'target' in all_features:
            mapped_features['target'] = final_model.mapping_target(all_features['target'])

        # Create inference blocks (use original graph)
        blocks = []
        for _ in range(args.num_layers):
            block = dgl.to_block(hetero_graph,
                                 dst_nodes={ntype: torch.arange(hetero_graph.num_nodes(ntype), device=device)
                                            for ntype in hetero_graph.ntypes})
            blocks.append(block)

        blocks[0].srcdata['features'] = mapped_features
        node_embeddings = final_model.HeteroGNN(blocks, mapped_features)

        # Evaluate positives and negatives
        if predictor_type in ['mlp', 'md']:
            pos_scores = predict_scores(final_model, node_embeddings, test_src, test_dst, device)
            neg_scores = predict_scores(final_model, node_embeddings, neg_src, neg_dst, device)
        else:
            node_embeddings = {k: v.to('cpu') for k, v in node_embeddings.items()}
            pos_scores = cos_sim(node_embeddings['ingredient'][test_src].numpy(),
                                 node_embeddings['target'][test_dst].numpy())
            neg_scores = cos_sim(node_embeddings['ingredient'][neg_src].numpy(),
                                 node_embeddings['target'][neg_dst].numpy())

    # Calculate test set performance
    pos_labels = np.ones(len(pos_scores))
    neg_labels = np.zeros(len(neg_scores))

    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([pos_labels, neg_labels])

    test_auroc = roc_auc_score(all_labels, all_scores)
    test_auprc = average_precision_score(all_labels, all_scores)

    optimal_threshold, optimal_f1 = find_optimal_threshold(all_scores, all_labels)
    predictions = (all_scores >= optimal_threshold).astype(int)
    test_f1 = f1_score(all_labels, predictions)

    # Print final results
    print("\n" + "=" * 60)
    print("BindingDB Fast Mode Evaluation Results:")
    print("=" * 60)
    print(f"Training Time: {training_time:.2f}s")
    print(f"\nTest Set Performance:")
    print(f"  AUROC: {test_auroc:.4f}")
    print(f"  AUPRC: {test_auprc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  Optimal Threshold: {optimal_threshold:.3f}")
    print(f"  Test Positives: {len(pos_scores)}, Negatives: {len(neg_scores)}")
    print("=" * 60)

    # Save model (if needed)
    if args.save_model:
        os.makedirs('saved_models', exist_ok=True)
        model_filename = get_model_filename(args.dataset, has_regularization, skip_cv=True)
        save_path = os.path.join('saved_models', model_filename)

        save_dict = {
            'model_state_dict': final_model.state_dict(),
            'args': args,
            'dataset': 'bindingdb',
            'mode': 'direct_training',
            'training_time': training_time,
            'test_performance': {
                'auroc': test_auroc,
                'auprc': test_auprc,
                'f1': test_f1,
                'threshold': optimal_threshold
            },
            'test_set_size': len(test_src)
        }
        torch.save(save_dict, save_path)
        print(f"\nModel saved to: {save_path}")

    # Visualization after training (BindingDB Direct Training Mode)
    if args.visualize_profile == 3:
        from visualize_embeddings2 import run_simple_ingredient_target_visualization
        print("\nRunning post-training Ingredient-Target Visualization (Mode 3)...")
        initial_features_dict = {
            'pretrained': hetero_graph.ndata['features']
        }
        # Note: BindingDB direct training mode currently does not support pre_training_embeddings
        run_simple_ingredient_target_visualization(args, hetero_graph, initial_features_dict,
                                                   model=model,
                                                   pre_training_embeddings=None)
    elif args.visualize_profile == 2:
        from visualize_embeddings2 import visualize_after_training
        print("\nRunning post-training Label Visualization (Mode 2)...")
        initial_features_dict = {
            'pretrained': hetero_graph.ndata['features']
        }
        visualize_after_training(model, hetero_graph, initial_features_dict, args)
    elif args.visualize_profile == 1:
        from visualize_embeddings import run_ligand_profile_analysis
        print("\nRunning post-training Ligand Profile Analysis (Mode 1)...")
        initial_features_dict = {
            'pretrained': hetero_graph.ndata['features']
        }
        run_ligand_profile_analysis(args, hetero_graph, initial_features_dict, model=model,
                                    filter_all_isolated=args.filter_all_isolated)
    return test_auroc, test_auprc, test_f1


def train_low_resource(args, hetero_graph, rel_list, device, predictor_type="md"):
    """
    Low Resource Learning Curve Experiment (Improved): Fixed Test Set + Sampled Training Set

    Process:
    1. Split 20% as fixed test set
    2. Sample 80% training set at different ratios
    3. Perform K-Fold CV on sampled data
    4. Evaluate final performance on fixed test set
    """
    import json
    from datetime import datetime

    # Get all edges
    it_eids = hetero_graph.edges(etype='it', form='eid')
    total_edges = len(it_eids)

    # Collect all positive pairs
    src_all, dst_all = hetero_graph.edges(etype='it')
    all_positive_pairs = set(zip(src_all.cpu().numpy(), dst_all.cpu().numpy()))
    num_ingredients = hetero_graph.num_nodes('ingredient')
    num_targets = hetero_graph.num_nodes('target')

    print(f"\n{'=' * 80}")
    print("Low Resource Learning Experiment (Improved)")
    print(f"{'=' * 80}")
    print(f"Total Edges: {total_edges}")
    print(f"Nodes: {num_ingredients} ingredients, {num_targets} targets")

    # ========== Step 1: Split Fixed Test Set (20%) ==========
    print("\nStep 1: Splitting fixed test set...")
    np.random.seed(42)  # Fixed seed for consistent test set

    test_ratio = 0.2
    test_size = int(total_edges * test_ratio)
    test_indices = np.random.choice(total_edges, test_size, replace=False)
    train_indices = np.setdiff1d(np.arange(total_edges), test_indices)

    # Get test set edges
    test_edges = it_eids[test_indices]
    train_edges_pool = it_eids[train_indices]  # Training candidate pool

    print(f"Fixed Test Set: {len(test_edges)} edges ({test_ratio * 100:.0f}%)")
    print(f"Train Pool: {len(train_edges_pool)} edges ({(1 - test_ratio) * 100:.0f}%)")

    # Prepare test set positives and negatives
    test_src = src_all[test_indices].cpu().numpy()
    test_dst = dst_all[test_indices].cpu().numpy()

    # Generate test set negative samples
    print("Generating test set negative samples...")
    test_neg_src, test_neg_dst = generate_true_negative_samples(
        all_positive_pairs, len(test_src), num_ingredients, num_targets
    )

    # Store all results
    all_results = {}

    # ========== Step 2: Sampling Experiment on Training Set ==========
    for ratio in args.sample_ratios:
        print(f"\n{'=' * 60}")
        print(f"Training Set Sampling Ratio: {ratio * 100:.1f}%")
        print(f"{'=' * 60}")

        ratio_results = {
            'cv_scores': [],  # CV scores
            'test_scores': []  # Test set scores
        }

        # Calculate actual sample size (based on train pool)
        sampled_size = max(10, int(len(train_edges_pool) * ratio))

        # Repeated experiments
        for repeat in range(args.low_resource_repeats):
            repeat_seed = 411 + repeat * 100
            set_seed(repeat_seed)

            print(f"\nRepeat {repeat + 1}/{args.low_resource_repeats} (seed={repeat_seed})")

            # Sample from training pool
            if ratio < 1.0:
                sampled_train_indices = np.random.choice(
                    len(train_edges_pool), sampled_size, replace=False
                )
                sampled_edges = train_edges_pool[sampled_train_indices]
            else:
                sampled_edges = train_edges_pool

            print(f"Sampled Training Edges: {len(sampled_edges)} / {len(train_edges_pool)}")

            # Create sampled training subgraph
            sampled_graph = create_sampled_subgraph(hetero_graph, sampled_edges, device)

            # ========== K-Fold CV ==========
            kf = KFold(n_splits=min(args.k_fold, len(sampled_edges)),
                       shuffle=True, random_state=repeat_seed)

            cv_scores = []
            best_model = None
            best_cv_score = -1

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(sampled_edges)):
                # Create training graph
                train_graph = create_train_graph_without_val_edges(
                    sampled_graph, train_idx, val_idx, device
                )

                # Prepare validation set
                sampled_src, sampled_dst = sampled_graph.edges(etype='it')
                val_src = sampled_src[val_idx].cpu().numpy()
                val_dst = sampled_dst[val_idx].cpu().numpy()

                val_neg_src, val_neg_dst = generate_true_negative_samples(
                    all_positive_pairs, len(val_src), num_ingredients, num_targets
                )

                # Create and train model
                model = Model(args, rel_list, predictor_type=predictor_type).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
                lr_sche = torch.optim.lr_scheduler.StepLR(opt, args.lr_period, args.lr_decay)

                # Prepare dataloader
                train_eid_dict = {
                    etype: train_graph.edges(etype=etype, form='eid')
                    for etype in train_graph.canonical_etypes
                }

                train_negative_sampler = GlobalUniform(args.k)
                train_sampler = dgl.dataloading.NeighborSampler([args.k] * args.num_layers)
                train_sampler = dgl.dataloading.as_edge_prediction_sampler(
                    train_sampler, negative_sampler=train_negative_sampler
                )

                train_dataloader = dgl.dataloading.DataLoader(
                    train_graph, train_eid_dict, train_sampler,
                    device=device, batch_size=args.batch_size,
                    shuffle=True, drop_last=False, num_workers=0
                )

                # Train model
                for epoch in range(args.num_epochs):
                    model.train()
                    for step, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(train_dataloader):
                        input_features = blocks[0].srcdata['features']
                        pos_score, neg_score, reg_terms = model(
                            args, positive_graph, negative_graph, blocks, input_features
                        )
                        loss = compute_loss(pos_score, neg_score, rel_list[0])

                        # Add regularization
                        lambda_div = getattr(args, 'lambda_head_div', 0.0)
                        lambda_ent = getattr(args, 'lambda_head_entropy', 0.0)
                        if lambda_div != 0 or lambda_ent != 0:
                            loss = loss + lambda_div * reg_terms['div'] + lambda_ent * reg_terms['ent']

                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    lr_sche.step()

                # Validation Evaluation
                model.eval()
                with torch.no_grad():
                    # Use full graph features
                    all_features = hetero_graph.ndata['features']

                    mapped_features = {}
                    if 'herb' in all_features and hasattr(model, 'mapping_herb'):
                        mapped_features['herb'] = model.mapping_herb(all_features['herb'])
                    if 'ingredient' in all_features:
                        mapped_features['ingredient'] = model.mapping_ingredient(all_features['ingredient'])
                    if 'target' in all_features:
                        mapped_features['target'] = model.mapping_target(all_features['target'])

                    # Create inference blocks
                    blocks = []
                    for _ in range(args.num_layers):
                        block = dgl.to_block(
                            hetero_graph,
                            dst_nodes={
                                ntype: torch.arange(hetero_graph.num_nodes(ntype), device=device)
                                for ntype in hetero_graph.ntypes
                            }
                        )
                        blocks.append(block)

                    blocks[0].srcdata['features'] = mapped_features
                    node_embeddings = model.HeteroGNN(blocks, mapped_features)

                    # Calculate validation scores
                    if predictor_type in ['mlp', 'md']:
                        positive_res = predict_scores(model, node_embeddings, val_src, val_dst, device)
                        negative_res = predict_scores(model, node_embeddings, val_neg_src, val_neg_dst, device)
                    else:
                        node_embeddings = {k: v.to('cpu') for k, v in node_embeddings.items()}
                        positive_res = cos_sim(np.array(node_embeddings['ingredient'][val_src]),
                                               np.array(node_embeddings['target'][val_dst]))
                        negative_res = cos_sim(np.array(node_embeddings['ingredient'][val_neg_src]),
                                               np.array(node_embeddings['target'][val_neg_dst]))

                # Calculate validation metrics
                positive_labels = np.ones(len(positive_res))
                negative_labels = np.zeros(len(negative_res))
                all_scores = np.concatenate([positive_res, negative_res])
                all_labels = np.concatenate([positive_labels, negative_labels])

                auroc = roc_auc_score(all_labels, all_scores)
                auprc = average_precision_score(all_labels, all_scores)

                cv_scores.append((auroc, auprc))

                # Save best model
                if auroc > best_cv_score:
                    best_cv_score = auroc
                    best_model = model

            # Calculate CV average scores
            cv_scores_arr = np.array(cv_scores)
            cv_mean_auroc = np.mean(cv_scores_arr[:, 0])
            cv_mean_auprc = np.mean(cv_scores_arr[:, 1])

            print(f"  CV Average: AUROC={cv_mean_auroc:.4f}, AUPRC={cv_mean_auprc:.4f}")

            # ========== Evaluate on Fixed Test Set ==========
            print(f"  Evaluating on fixed test set...")

            best_model.eval()
            with torch.no_grad():
                all_features = hetero_graph.ndata['features']

                mapped_features = {}
                if 'herb' in all_features and hasattr(best_model, 'mapping_herb'):
                    mapped_features['herb'] = best_model.mapping_herb(all_features['herb'])
                if 'ingredient' in all_features:
                    mapped_features['ingredient'] = best_model.mapping_ingredient(all_features['ingredient'])
                if 'target' in all_features:
                    mapped_features['target'] = best_model.mapping_target(all_features['target'])

                blocks = []
                for _ in range(args.num_layers):
                    block = dgl.to_block(
                        hetero_graph,
                        dst_nodes={
                            ntype: torch.arange(hetero_graph.num_nodes(ntype), device=device)
                            for ntype in hetero_graph.ntypes
                        }
                    )
                    blocks.append(block)

                blocks[0].srcdata['features'] = mapped_features
                node_embeddings = best_model.HeteroGNN(blocks, mapped_features)

                # Calculate test set scores
                if predictor_type in ['mlp', 'md']:
                    test_pos_res = predict_scores(best_model, node_embeddings, test_src, test_dst, device)
                    test_neg_res = predict_scores(best_model, node_embeddings, test_neg_src, test_neg_dst, device)
                else:
                    node_embeddings = {k: v.to('cpu') for k, v in node_embeddings.items()}
                    test_pos_res = cos_sim(np.array(node_embeddings['ingredient'][test_src]),
                                           np.array(node_embeddings['target'][test_dst]))
                    test_neg_res = cos_sim(np.array(node_embeddings['ingredient'][test_neg_src]),
                                           np.array(node_embeddings['target'][test_neg_dst]))

            # Calculate test metrics
            test_pos_labels = np.ones(len(test_pos_res))
            test_neg_labels = np.zeros(len(test_neg_res))
            test_all_scores = np.concatenate([test_pos_res, test_neg_res])
            test_all_labels = np.concatenate([test_pos_labels, test_neg_labels])

            test_auroc = roc_auc_score(test_all_labels, test_all_scores)
            test_auprc = average_precision_score(test_all_labels, test_all_scores)

            print(f"  Test Set: AUROC={test_auroc:.4f}, AUPRC={test_auprc:.4f}")

            # Store results
            ratio_results['cv_scores'].append({'auroc': cv_mean_auroc, 'auprc': cv_mean_auprc})
            ratio_results['test_scores'].append({'auroc': test_auroc, 'auprc': test_auprc})

        # Calculate statistics for this ratio
        cv_aurocs = [r['auroc'] for r in ratio_results['cv_scores']]
        cv_auprcs = [r['auprc'] for r in ratio_results['cv_scores']]
        test_aurocs = [r['auroc'] for r in ratio_results['test_scores']]
        test_auprcs = [r['auprc'] for r in ratio_results['test_scores']]

        all_results[ratio] = {
            'cv_auroc_mean': np.mean(cv_aurocs),
            'cv_auroc_std': np.std(cv_aurocs),
            'cv_auprc_mean': np.mean(cv_auprcs),
            'cv_auprc_std': np.std(cv_auprcs),
            'test_auroc_mean': np.mean(test_aurocs),
            'test_auroc_std': np.std(test_aurocs),
            'test_auprc_mean': np.mean(test_auprcs),
            'test_auprc_std': np.std(test_auprcs),
            'details': ratio_results
        }

    # Print final summary
    print(f"\n{'=' * 100}")
    print("Low Resource Learning Experiment Summary")
    print(f"Feature Type: {'Baseline (ECFP/PseAAC)' if args.herb_use_baseline else 'Pretrained (MolFormer/ESM)'}")
    print(f"Fixed Test Set Size: {len(test_edges)} edges")
    print(f"{'=' * 100}")
    print(f"{'Ratio':>8} | {'CV AUROC':>16} | {'CV AUPRC':>16} | {'Test AUROC':>16} | {'Test AUPRC':>16}")
    print("-" * 100)

    for ratio in args.sample_ratios:
        res = all_results[ratio]
        print(f"{ratio * 100:>7.1f}% | "
              f"{res['cv_auroc_mean']:.4f}±{res['cv_auroc_std']:.4f} | "
              f"{res['cv_auprc_mean']:.4f}±{res['cv_auprc_std']:.4f} | "
              f"{res['test_auroc_mean']:.4f}±{res['test_auroc_std']:.4f} | "
              f"{res['test_auprc_mean']:.4f}±{res['test_auprc_std']:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_type = "baseline" if args.herb_use_baseline else "pretrained"
    result_file = f"low_resource_fixed_test_{args.dataset}_{feature_type}_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'feature_type': feature_type,
            'test_ratio': test_ratio,
            'test_size': len(test_edges),
            'sample_ratios': args.sample_ratios,
            'repeats': args.low_resource_repeats,
            'k_fold': args.k_fold,
            'results': all_results
        }, f, indent=2)

    print(f"\nResults saved to: {result_file}")

    return all_results


def create_sampled_subgraph(hetero_graph, sampled_edge_ids, device):
    """
    Create subgraph based on sampled edges
    """
    # Get source and target nodes of sampled edges
    src_all, dst_all = hetero_graph.edges(etype='it')

    # Create sample mask
    sample_mask = torch.zeros(len(src_all), dtype=torch.bool)
    sample_mask[sampled_edge_ids] = True

    # Get sampled edges
    sampled_src = src_all[sample_mask]
    sampled_dst = dst_all[sample_mask]

    # Construct new graph data
    graph_data = {
        ('ingredient', 'it', 'target'): (sampled_src, sampled_dst),
        ('target', 'ti', 'ingredient'): (sampled_dst, sampled_src)
    }

    # Add other edge types (similarity edges etc.)
    for etype in hetero_graph.canonical_etypes:
        src_type, edge_type, dst_type = etype
        if edge_type not in ['it', 'ti']:
            src, dst = hetero_graph.edges(etype=etype)
            graph_data[etype] = (src, dst)

    # Get node counts
    num_nodes_dict = {ntype: hetero_graph.num_nodes(ntype) for ntype in hetero_graph.ntypes}

    # Create new graph
    sampled_graph = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

    # Copy node features
    for ntype in hetero_graph.ntypes:
        if 'features' in hetero_graph.nodes[ntype].data:
            sampled_graph.nodes[ntype].data['features'] = hetero_graph.nodes[ntype].data['features']

    return sampled_graph.to(device)