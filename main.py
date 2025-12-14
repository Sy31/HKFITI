import torch
import argparse
from utils import build_graph, process_data, process_drugbank_data, process_davis_data, process_bindingdb_data
from train import train, unseen_train, train_bindingdb_with_test, train_bindingdb_direct
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings

warnings.filterwarnings('ignore')
# ---- argparse BooleanOptionalAction shim (Py3.8 compatibility) ----
try:
    BooleanOptionalAction = argparse.BooleanOptionalAction  # Py3.9+
except AttributeError:
    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, **kwargs):
            # Supports --flag / --no-flag options
            _option_strings = []
            for opt in option_strings:
                _option_strings.append(opt)
                if opt.startswith('--'):
                    _option_strings.append('--no-' + opt[2:])
            super().__init__(_option_strings, dest, nargs=0, default=default, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            if option_string and option_string.startswith('--no-'):
                setattr(namespace, self.dest, False)
            else:
                setattr(namespace, self.dest, True)
# ---------------------------------------------------------


warnings.filterwarnings('ignore')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Train GNN for link prediction on herb-ingredient-target or drug-target network.')

    # Dataset selection parameters
    parser.add_argument('--dataset', default='herb', type=str,
                        choices=['herb', 'drugbank', 'davis', 'bindingdb'],
                        help='Choose dataset: herb, drugbank, davis, or bindingdb')

    # ============= Herb Dataset Baseline Vector Parameters =============
    parser.add_argument('--herb_use_baseline', type=str, default='ME',
                        choices=['ME', 'EP', 'MV', 'BG'],
                        help='(Herb only) Feature type: ME(MolFormer&ESM-2), EP(ECFP&PseAAC), MV(Metapath2Vec), BG(Bipartite Graph)')
    parser.add_argument('--herb_baseline_dims', nargs=2, type=int, default=[128, 128],
                        help='(Herb only) Baseline vector dims [ingredient_dim, target_dim], default [128, 128]')
    # EP 2048, 50 MV 128,128 BG 128,128

    # ============= Low Resource Learning Parameters =============
    parser.add_argument('--low_resource_mode', action=BooleanOptionalAction, default=False,
                        help='Enable low-resource learning curve experiments')
    parser.add_argument('--sample_ratios', nargs='+', type=float,
                        default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.25, 0.5, 0.75, 1],
                        help='List of data sampling ratios')
    parser.add_argument('--low_resource_repeats', default=3, type=int,
                        help='Number of repeats per sampling ratio (using different random seeds)')

    # ========== Visualization Parameters ==========
    parser.add_argument('--visualize_profile', type=int, default=0,
                        choices=[0, 1],
                        help='Vis mode: 0-None, 1-Ligand Profile')
    parser.add_argument('--clustering_method', type=str, default='eigengap',
                        choices=['eigengap', 'silhouette'],
                        help='Clustering metric: eigengap or silhouette')
    parser.add_argument('--filter_all_isolated', action=BooleanOptionalAction, default=False,
                        help='Isolate isolated nodes (ME/EP). MV, BG are always isolated')

    # Training Hyperparameters
    parser.add_argument('--num_epochs', default=20, type=int, help='Maximum training epochs')
    parser.add_argument('--lr_period', default=4, type=int, help='LR scheduler period (epochs)')
    parser.add_argument('--lr', default=2e-4, type=float, help='Learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--lr_decay', default=0.8, type=float,
                        help='Learning rate decay factor')
    # herb 20 4 2e-4 1e-4 0.8 5000 k=7
    # drugbank 30 3 4e-4 1e-4 0.9 5000 k=7
    # bindingDB 20 4 4e-4 2e-4 0.9 10000 k=7

    # Embedding Dimensions
    parser.add_argument('--herb_in_dim', default=23, type=int, help='Herb embedding dimension (Herb dataset only)')
    parser.add_argument('--ingredient_in_dim', default=768, type=int, help='Ingredient/Drug embedding dimension')
    parser.add_argument('--target_in_dim', default=1280, type=int, help='Target embedding dimension')

    parser.add_argument('--in_dim', default=512, type=int, help='Input embedding dimension')
    parser.add_argument('--h_dim', default=256, type=int, help='Hidden embedding dimension')
    parser.add_argument('--out_dim', default=64, type=int, help='Output embedding dimension')
    parser.add_argument('--num_heads', default=4, type=int, help='Number of attention heads [1,2,4,8,16]')

    # Model Architecture Flags
    parser.add_argument('--use_simple_gnn', action=BooleanOptionalAction, default=False, help='Use simple GNN')
    parser.add_argument('--use_attention', action=BooleanOptionalAction, default=True, help='Whether to use attention mechanism')

    parser.add_argument('--cuda', default=0, type=int, help='GPU index')
    parser.add_argument('--k', default=7, type=int, help='Negative sampling ratio (negatives per positive)')
    parser.add_argument('--batch_size', default=5000, type=int, help='Batch size')
    parser.add_argument('--num_layers', default=2, type=int, help='Number of GNN layers')
    parser.add_argument('--k_fold', default=10, type=int, help='K-fold cross-validation')

    parser.add_argument('--unseen_setting', default=0, type=int, help='0: Full network; 2: Cold target setting')
    parser.add_argument('--graph_struct', default=3, type=int,
                        help='0: No similarity; 1: Ingredient sim only; 2: Target sim only; 3: Both')

    parser.add_argument('--remove_herb', default=True, action='store_true',
                        help='Whether to remove herb nodes (Herb dataset only)')
    parser.add_argument('--use_herb_zero_aware', action=BooleanOptionalAction, default=True,
                        help='Use zero-aware handling for sparse herb vectors (Herb dataset only)')
    parser.add_argument('--herb_encoder_dropout', default=0.1, type=float, help='Dropout rate in HerbEncoder')
    parser.add_argument('--zero_importance_init', default=0.5, type=float, help='Initial value for zero-value importance')

    parser.add_argument('--save_model', action=BooleanOptionalAction, default=False, help='Whether to save the trained model')
    parser.add_argument('--save_fold1_model', default=False, action='store_true', help='Whether to save the model of the first fold')
    parser.add_argument('--skip_cv', default=False, action='store_true', help='Skip K-fold CV (BindingDB only)')

    # Predictor Parameters
    parser.add_argument('--predictor_type', type=str, default='md', choices=['mlp', 'dot', 'md'],
                        help='Predictor type: mlp, dot, md (mlp&dot)')
    parser.add_argument('--mlp_hidden_dims', nargs='+', type=int, default=[128, 64, 32])
    parser.add_argument('--mlp_dropout', default=0.1, type=float, help='Dropout rate in MLP predictor')

    # Mixed Predictor Parameters
    parser.add_argument('--fusion_method', type=str, default='gated', choices=['residual', 'weighted', 'gated-mix'],
                        help='Fusion method: residual, weighted, gated-mix')
    parser.add_argument('--learnable_weight', action=BooleanOptionalAction, default=False,
                        help='Use learnable fusion weights (weighted only)')
    parser.add_argument('--residual_scale', default=1.0, type=float,
                        help='Scaling factor for residual term (residual only)')

    # Structural Noise Parameters
    parser.add_argument('--edge_noise_ratio', default=0, type=float, help='Structural redundancy edge noise ratio')
    parser.add_argument('--edge_noise_etype', default='it', type=str, help='Edge type for edge noise')
    parser.add_argument('--edge_noise_dis', nargs=5, type=float, default=[0, 0, 1, 0, 0],
                        help='Edge noise distribution [random, cluster, feature, multi-modal, reserved]')

    # Feature Gaussian Noise Parameters
    parser.add_argument('--feature_noise_std', default=0, type=float,
                        help='Feature Gaussian noise std dev (0 means disabled)')
    parser.add_argument('--feature_noise_types', nargs='+', default=['herb', 'ingredient', 'target'],
                        help='Node types to add feature noise (herb ingredient target)')

    # Regularization Parameters 0.1, 0.02 for feature Gaussian noise; 0.03, 0.1 for cold-start
    parser.add_argument('--lambda_head_div', default=0, type=float)
    parser.add_argument('--lambda_head_entropy', default=0, type=float)

    parser.add_argument('--layer_reg_decay', default=0.8, type=float, help='Layer regularization weight decay')
    parser.add_argument('--reg_ntypes', nargs='+', default=['drug', 'ingredient', 'target'])
    parser.add_argument('--p_headdrop', default=0, type=float, help='Head dropout probability (valid only with attention)')


    args = parser.parse_args()

    # Automatically adjust parameters based on dataset
    if args.dataset in ['drugbank', 'davis', 'bindingdb']:
        args.remove_herb = True
        args.reg_ntypes = ['ingredient', 'target']
        print(f"\nUsing {args.dataset.upper()} dataset, automatically setting remove_herb=True (no herb nodes)")

    return args


def print_config(args):
    """Print configuration information."""
    print(f"\n================== Configuration ==================")
    print(f"Dataset: {args.dataset}")
    # ============= Display Herb Dataset Vector Type =============
    if args.dataset == 'herb':
        if args.herb_use_baseline == 'ME':
            print(f"Feature Vector: Pretrained (MolFormer/ESM-2)")
        elif args.herb_use_baseline == 'EP':
            print(f"Feature Vector: Baseline (ECFP/PseAAC)")
            print(f"  Ingredient Dim: {args.herb_baseline_dims[0]} (ECFP)")
            print(f"  Target Dim: {args.herb_baseline_dims[1]} (PseAAC)")
        elif args.herb_use_baseline == 'MV':
            print(f"Feature Vector: Metapath2Vec")
        elif args.herb_use_baseline == 'BG':
            print(f"Feature Vector: Bipartite Graph")
    print(f"Epochs (num_epochs): {args.num_epochs}")
    print(f"K-Fold (k_fold): {args.k_fold}")
    print(f"Attention Heads (num_heads): {args.num_heads}")
    print(f"GNN Layers (num_layers): {args.num_layers}")
    print(f"Negative Sampling (k): {args.k}")
    print(f"Batch Size (batch_size): {args.batch_size}")
    print(f"Use Attention: {'Yes' if getattr(args, 'use_attention', True) else 'No (Simple GNN)'}")

    # Feature/Structure Noise Info
    if getattr(args, 'edge_noise_ratio', 0) != 0:
        dis = getattr(args, 'edge_noise_dis', [])
        if isinstance(dis, (list, tuple)) and len(dis) >= 4:
            print(
                f"Edge Noise Dist: Random={dis[0]:.2f}, Cluster={dis[1]:.2f}, Feature={dis[2]:.2f}, Multi-modal={dis[3]:.2f}")
        else:
            print("Edge Noise Dist: Invalid parameter format, detailed distribution not printed")

    # Feature Noise Configuration
    if getattr(args, 'feature_noise_std', 0) != 0:
        print(f"Feature Noise Std (feature_noise_std): {args.feature_noise_std}")
        print(f"Feature Noise Types (feature_noise_types): {args.feature_noise_types}")

    print(f"Head Diversity Reg: {args.lambda_head_div}")
    print(f"Head Entropy Reg: {args.lambda_head_entropy}")

    # Graph Structure Description
    graph_struct_desc = {
        0: "No similarity network",
        1: "Ingredient similarity only",
        2: "Target similarity only",
        3: "Ingredient + Target similarity"
    }
    print(f"Graph Structure (graph_struct): {args.graph_struct} - {graph_struct_desc.get(args.graph_struct, 'Unknown')}")

    # Display Herb Node Removal Status
    if args.dataset == 'herb':
        print(f"Remove Herb Nodes (remove_herb): {'Yes' if args.remove_herb else 'No'}")
        if not args.remove_herb:
            if args.use_herb_zero_aware:
                print("Use Hybrid Herb Encoder: Yes (Zero-Aware)")
                print(f"  Herb Encoder Dropout: {args.herb_encoder_dropout}")
                print(f"  Zero Importance Init: {args.zero_importance_init}")
            else:
                print("Use Hybrid Herb Encoder: No (Standard MLP)")

    # Display Skip CV (BindingDB Only)
    if args.dataset == 'bindingdb' and hasattr(args, 'skip_cv'):
        print(f"Skip Cross-Validation (skip_cv): {'Yes' if args.skip_cv else 'No'}")

    unseen_desc = {0: "(Full Network)", 2: "(Cold Target)"}
    print(f"Unseen Setting (unseen_setting): {args.unseen_setting} - {unseen_desc.get(args.unseen_setting, 'Unknown')}")
    print(f"===============================================\n")


def main(args):
    device = f'cuda:{args.cuda}' if args.cuda >= 0 and torch.cuda.is_available() else 'cpu'

    # Handle BindingDB Dataset
    if args.dataset == 'bindingdb':
        print("\n==================== Loading BindingDB Dataset ==================")
        result = process_bindingdb_data()

        if len(result) == 5:  # Has test set
            it_edges, is_edges, ts_edges, initial_features, test_edges = result
            hi_edges = None

            # Set node counts for BindingDB
            num_nodes_dict = {
                'ingredient': initial_features['pretrained']['ingredient'].shape[0],
                'target': initial_features['pretrained']['target'].shape[0]
            }

            print(f"\nBindingDB Node Counts:")
            print(f"  Ingredient Nodes: {num_nodes_dict['ingredient']}")
            print(f"  Target Nodes: {num_nodes_dict['target']}")

            # Build Graph
            hetero_graph, rel_list = build_graph(args, it_edges, is_edges, ts_edges,
                                                 hi_edges, initial_features, device,
                                                 num_nodes_dict=num_nodes_dict)

            # Print Config
            print_config(args)

            # Check if skipping cross-validation
            if args.skip_cv:
                print("\n========== Skipping Cross-Validation Mode ==========")
                print("Directly training final model and evaluating on test set")
                print("=" * 40)

                train_bindingdb_direct(args, hetero_graph, rel_list, test_edges,
                                       device, predictor_type=args.predictor_type)
            else:
                # Use standard three-stage process (including CV)
                if args.unseen_setting == 0:
                    train_bindingdb_with_test(args, hetero_graph, rel_list, test_edges,
                                              device, predictor_type=args.predictor_type)
                else:
                    print("Warning: Unseen setting not supported in BindingDB test set mode")

        else:  # No test set, use regular process
            it_edges, is_edges, ts_edges, initial_features = result
            hi_edges = None

            num_nodes_dict = {
                'ingredient': initial_features['pretrained']['ingredient'].shape[0],
                'target': initial_features['pretrained']['target'].shape[0]
            }

            hetero_graph, rel_list = build_graph(args, it_edges, is_edges, ts_edges,
                                                 hi_edges, initial_features, device,
                                                 num_nodes_dict=num_nodes_dict)
            print_config(args)

            if args.unseen_setting == 0:
                train(args, hetero_graph, rel_list, device, predictor_type=args.predictor_type)
            else:
                unseen_train(args, hetero_graph, rel_list, device, predictor_type=args.predictor_type)

    # Handle other datasets
    else:
        # Ignore skip_cv for other datasets
        if hasattr(args, 'skip_cv') and args.skip_cv:
            print(f"\nNote: --skip_cv parameter is only valid for BindingDB, ignored for {args.dataset}")

        # Choose data processing function based on dataset
        if args.dataset == 'drugbank':
            print("\n==================== Loading DrugBank Dataset ==================")
            it_edges, is_edges, ts_edges, initial_features = process_drugbank_data()
            hi_edges = None
        elif args.dataset == 'davis':
            print("\n==================== Loading Davis Dataset ==================")
            it_edges, is_edges, ts_edges, initial_features = process_davis_data()
            hi_edges = None
        else:  # herb
            print("\n==================== Loading Herb Dataset ==================")
            # ============= Herb Dataset Baseline Vector Logic =============
            it_edges, is_edges, ts_edges, hi_edges, initial_features = process_data(
                vector_type=args.herb_use_baseline,
                baseline_dims=args.herb_baseline_dims
            )

            # Update input dimensions based on vector type
            if args.herb_use_baseline == 'EP':
                args.ingredient_in_dim = args.herb_baseline_dims[0]  # ECFP dim
                args.target_in_dim = args.herb_baseline_dims[1]  # PseAAC dim
                print(f"Updated input dims: ingredient={args.ingredient_in_dim}, target={args.target_in_dim}")
            elif args.herb_use_baseline == 'MV' or args.herb_use_baseline == 'BG':
                # Metapath2Vec dims obtained from actual data
                args.ingredient_in_dim = initial_features['pretrained']['ingredient'].shape[1]
                args.target_in_dim = initial_features['pretrained']['target'].shape[1]
                print(f"Metapath2Vec/BG actual dims: ingredient={args.ingredient_in_dim}, target={args.target_in_dim}")
            # =============================================================

        # Build Graph
        hetero_graph, rel_list = build_graph(args, it_edges, is_edges, ts_edges,
                                             hi_edges, initial_features, device)

        # ========== Pre-training Visualization Analysis ==========
        if args.visualize_profile == 1:
            # Mode 1: Ligand Profile Analysis
            from visualize_embeddings import run_ligand_profile_analysis
            print("\nPerforming pre-training ligand profile analysis...")
            run_ligand_profile_analysis(args, hetero_graph, initial_features,
                                        filter_all_isolated=args.filter_all_isolated)
        elif args.visualize_profile == 2:
            # Mode 2: Label Visualization
            from visualize_embeddings2 import run_label_visualization
            print("\nPerforming label visualization...")
            run_label_visualization(args, hetero_graph, initial_features)
        elif args.visualize_profile in [3, 4]:
            # Mode 3: Simple Ingredient-Target Visualization
            from visualize_embeddings2 import run_simple_ingredient_target_visualization
            print("\nPreparing Ingredient-Target visualization data...")
            # Pre-training call: only prepare data, do not plot
            pre_training_embeddings = run_simple_ingredient_target_visualization(args, hetero_graph, initial_features)
        elif args.visualize_profile == 4:
            # Mode 4: t-SNE version of Ingredient-Target Visualization
            from visualize_embeddings2 import run_tsne_ingredient_target_visualization
            print("\nPreparing Ingredient-Target t-SNE visualization data...")
            # Pre-training call: only prepare data, do not plot
            pre_training_embeddings = run_tsne_ingredient_target_visualization(args, hetero_graph, initial_features)
        else:
            pre_training_embeddings = None
        # Print Config
        print_config(args)

        # Train Model
        if args.low_resource_mode:
            # Low Resource Learning Mode
            from train import train_low_resource
            print("\n========== Low Resource Learning Experiment Mode ==========")
            print(f"Sampling Ratios: {args.sample_ratios}")
            print(f"Repeats: {args.low_resource_repeats}")
            print(
                f"Feature Type: { {'ME': 'Pretrained(MolFormer/ESM-2)', 'EP': 'Baseline(ECFP/PseAAC)', 'BG': 'Bipartite Graph', 'MV': 'Metapath2Vec', }[args.herb_use_baseline]}")
            print("=" * 40)
            train_low_resource(args, hetero_graph, rel_list, device,
                               predictor_type=args.predictor_type)
        else:
            # Normal Training Mode
            if args.unseen_setting == 0:
                train(args, hetero_graph, rel_list, device,
                      predictor_type=args.predictor_type,
                      pre_training_embeddings=pre_training_embeddings if args.visualize_profile in [3, 4] else None)
            else:
                unseen_train(args, hetero_graph, rel_list, device,
                             predictor_type=args.predictor_type,
                             pre_training_embeddings=pre_training_embeddings if args.visualize_profile in [3,
                                                                                                           4] else None)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)