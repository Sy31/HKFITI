import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
from torch import nn
from dgl import function as fn
import torch
import dgl



# ================== Utility Functions ==================
def _head_entropy_loss(x, eps=1e-9):
    """
    x: Tensor [B, H, D] -> Calculate entropy loss of head energy distribution (positive value, larger means less uniform)
    """
    B, H, D = x.shape
    energy = (x ** 2).sum(dim=-1)  # [B, H]
    z = energy.sum(dim=1, keepdim=True) + eps
    p = energy / z  # [B, H]
    loss = (p * (p.add(eps).log())).sum(dim=1).mean()
    return loss


def _head_decorrelation_loss(x, eps=1e-9):
    """
    x: Tensor [B, H, D] -> Decorrelation loss
    """
    B, H, D = x.shape
    Xh = x.permute(1, 0, 2).contiguous().view(H, B * D)  # [H, N]
    Xh = Xh - Xh.mean(dim=1, keepdim=True)
    Xh = Xh / (Xh.norm(p=2, dim=1, keepdim=True) + eps)
    C = Xh @ Xh.t()  # [H, H]
    off = C - torch.eye(H, device=x.device)
    loss = (off ** 2).sum() / (H * (H - 1) + eps)
    return loss


# ================== Simple GNN (Fully Homogeneous) ==================
class SimpleGNN(nn.Module):
    """True homogeneous GNN - parameters shared across all node types"""

    def __init__(self, in_feats, hid_feats, out_feats, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.drop_rate = 0.2

        # Single GNN layer, shared across all node types
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(dglnn.GraphConv(in_feats, out_feats, bias=True))
        else:
            # First layer
            self.convs.append(dglnn.GraphConv(in_feats, hid_feats, bias=True))
            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(dglnn.GraphConv(hid_feats, hid_feats, bias=True))
            # Output layer
            self.convs.append(dglnn.GraphConv(hid_feats, out_feats, bias=True))

        # Shared BatchNorm
        self.bn = nn.BatchNorm1d(hid_feats if num_layers > 1 else out_feats)

    def forward(self, blocks, inputs):
        """Convert heterogeneous graph to homogeneous and propagate"""
        device = blocks[0].device

        # Step 1: Collect initial features of all nodes
        # Create mapping from node type to global index
        node_offsets = {}
        total_nodes = 0
        for ntype in ['herb', 'ingredient', 'target']:
            if ntype in inputs:
                node_offsets[ntype] = total_nodes
                total_nodes += inputs[ntype].shape[0]

        # Merge all node features into a single tensor
        h = torch.zeros((total_nodes, inputs['ingredient'].shape[1]),
                        dtype=torch.float32, device=device)

        for ntype in ['herb', 'ingredient', 'target']:
            if ntype in inputs:
                start_idx = node_offsets[ntype]
                end_idx = start_idx + inputs[ntype].shape[0]
                h[start_idx:end_idx] = inputs[ntype]

        # Layer-wise propagation
        for l in range(self.num_layers):
            block = blocks[l]

            # Convert heterogeneous graph block to homogeneous
            # Collect all edges, disregarding types
            all_src = []
            all_dst = []

            for etype in block.canonical_etypes:
                src_type, _, dst_type = etype
                src, dst = block.edges(etype=etype)

                if len(src) > 0:
                    # Convert to global indices
                    global_src = src + node_offsets.get(src_type, 0)
                    global_dst = dst + node_offsets.get(dst_type, 0)
                    all_src.append(global_src)
                    all_dst.append(global_dst)

            if all_src:
                # Merge all edges
                all_src = torch.cat(all_src)
                all_dst = torch.cat(all_dst)

                # Create homogeneous graph
                homo_graph = dgl.graph((all_src, all_dst), num_nodes=total_nodes).to(device)

                # Add self-loops
                homo_graph = dgl.add_self_loop(homo_graph)

                # Apply GNN layer (parameters shared across all nodes)
                h = self.convs[l](homo_graph, h)

                # Activation and regularization (except for the last layer)
                if l < self.num_layers - 1:
                    h = self.bn(h)
                    h = F.leaky_relu(h)
                    h = F.dropout(h, p=self.drop_rate, training=self.training)

        # Split results back to each node type
        h_dict = {}
        for ntype in ['herb', 'ingredient', 'target']:
            if ntype in inputs:
                start_idx = node_offsets[ntype]
                end_idx = start_idx + inputs[ntype].shape[0]
                # Extract only target nodes (nodes in the last block)
                if ntype in blocks[-1].dsttypes:
                    num_dst = blocks[-1].number_of_dst_nodes(ntype)
                    h_dict[ntype] = h[start_idx:start_idx + num_dst]

        return h_dict

    def forward_with_headwise(self, blocks, inputs, head_mask=None):
        """Return same format as HeteroGNN for compatibility"""
        h = self.forward(blocks, inputs)
        # For SimpleGNN, create fake headwise format (1 head)
        headwise = {k: v.unsqueeze(1) for k, v in h.items()}
        return h, headwise, []


#Zero-aware Herb Encoder

class HerbEncoder(nn.Module):
    def __init__(self, input_dim=23, output_dim=512, dropout=0.1, zero_importance_init=0.5):
        super().__init__()

        self.zero_embeddings = nn.Parameter(torch.randn(input_dim, output_dim))
        self.zero_importance = nn.Parameter(torch.ones(input_dim) * zero_importance_init)

        # Keep bias - important for sparse features
        self.nonzero_projection = nn.Linear(input_dim, output_dim)

        self.fusion = nn.Linear(output_dim * 2, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Optional: Single gating mechanism
        self.gate = nn.Parameter(torch.tensor(0.0))  # 0.5 after sigmoid

    def forward(self, x):
        nonzero_mask = (x != 0).float()
        nonzero_features = self.nonzero_projection(x * nonzero_mask)

        zero_mask = (x == 0).float()
        weighted_zero_mask = zero_mask * torch.sigmoid(self.zero_importance)
        weighted_zero_mask = weighted_zero_mask / (weighted_zero_mask.sum(dim=1, keepdim=True) + 1e-8)

        zero_features = torch.matmul(
            weighted_zero_mask.unsqueeze(1),
            self.zero_embeddings.unsqueeze(0)
        ).squeeze(1)

        # Optional: Use gating instead of concatenation
        # g = torch.sigmoid(self.gate)
        # combined = g * nonzero_features + (1-g) * zero_features

        combined = torch.cat([nonzero_features, zero_features], dim=-1)
        output = self.fusion(combined)

        output = self.layer_norm(output)
        if self.training:
            output = self.dropout(output)

        return output


# Zero-aware TCM Attribute Encoder2
"""
class HerbEncoder(nn.Module):


    def __init__(self, input_dim=23, output_dim=512, dropout=0.1, zero_importance_init=0.5):
        super().__init__()

        # Zero-value semantic embeddings (E_zero in paper)
        self.zero_embeddings = nn.Parameter(torch.randn(input_dim, output_dim))
        nn.init.xavier_uniform_(self.zero_embeddings)  # Better initialization

        # Learnable importance weights for zero dimensions (ω in paper)
        self.zero_importance = nn.Parameter(torch.ones(input_dim) * zero_importance_init)

        # Linear projection for non-zero values (W_nz in paper)
        self.nonzero_projection = nn.Linear(input_dim, output_dim)

        # Feature fusion layers
        self.fusion = nn.Sequential(  # Add non-linearity
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        batch_size = x.size(0)

        # Process non-zero values (explicit features)
        nonzero_mask = (x != 0).float()  # m_nz in paper
        h_exist = self.nonzero_projection(x * nonzero_mask)

        # Process zero values (implicit features)  
        zero_mask = (x == 0).float()  # m_z in paper

        # Apply importance weighting with normalization
        weighted_zero_mask = zero_mask * torch.sigmoid(self.zero_importance)
        # Add dim check to avoid all-zero case
        mask_sum = weighted_zero_mask.sum(dim=1, keepdim=True)
        mask_sum = torch.clamp(mask_sum, min=1e-8)  # Prevent division by zero
        weighted_zero_mask = weighted_zero_mask / mask_sum

        # Clearer matrix multiplication, avoiding squeeze issues
        # [batch, input_dim] x [input_dim, output_dim] -> [batch, output_dim]
        h_absent = torch.matmul(weighted_zero_mask, self.zero_embeddings)

        # Fusion of explicit and implicit features
        combined = torch.cat([h_exist, h_absent], dim=-1)  # [batch, output_dim*2]
        output = self.fusion(combined)  # [batch, output_dim]

        # Final normalization and dropout
        output = self.layer_norm(output)
        if self.training:
            output = self.dropout(output)

        return output

    def get_feature_importance(self):

        #Get the learned importance of each TCM attribute dimension.
        with torch.no_grad():
            return torch.sigmoid(self.zero_importance).cpu().numpy()
"""


# Simple Aggregator (No Attention)
class SimpleAggregator(nn.Module):
    """Simple mean aggregator for standard GNN"""

    def __init__(self, rel_names):
        super().__init__()
        self.rel_names = sorted(list(rel_names))

    def forward(self, h_dict, node_type=None):
        """Simple mean aggregation of features from different relations"""
        if isinstance(h_dict, dict):
            # Get features from all relations
            tensors = []
            for r in self.rel_names:
                if r in h_dict:
                    tensors.append(h_dict[r])

            if len(tensors) == 0:
                tensors = list(h_dict.values())

            if len(tensors) == 1:
                return tensors[0]

            # Simple mean
            stacked = torch.stack(tensors, dim=0)
            return torch.mean(stacked, dim=0)

        elif isinstance(h_dict, list):
            if len(h_dict) == 0:
                raise ValueError("Empty inputs for aggregator.")
            if len(h_dict) == 1:
                return h_dict[0]

            # Simple mean
            stacked = torch.stack(h_dict, dim=0)
            return torch.mean(stacked, dim=0)

        else:
            raise TypeError(f"Unsupported inputs type for aggregator: {type(h_dict)}")


# RelationAttentionAggregator
class RelationAttentionAggregator(nn.Module):
    def __init__(self, rel_names, out_dim, num_heads=8, attn_drop=0.1, reg_ntypes=None):
        super().__init__()
        self.rel_names = sorted(list(rel_names))
        self.out_dim = out_dim
        self.num_heads = num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.head_dim = out_dim // num_heads

        self.score_mlp = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim, bias=True),
            nn.Tanh(),
            nn.Linear(self.head_dim, 1, bias=True)
        )
        self.dropout = nn.Dropout(attn_drop)

    def _to_BHD(self, x):
        """Unify [B, H, D] or [B, H*D] to [B, H, D]"""
        if x.dim() == 3 and x.size(1) == self.num_heads:
            return x  # [B, H, D]
        if x.dim() == 2 and x.size(-1) == self.out_dim:
            B = x.size(0)
            return x.view(B, self.num_heads, self.head_dim)  # [B, H, D]
        raise ValueError(f"Unexpected tensor shape {tuple(x.shape)}")

    def forward(self, h_dict, node_type=None):
        # Unify to tensors:list[Tensor[B,H,D]]
        tensors = None
        if isinstance(h_dict, list):
            if len(h_dict) == 0:
                raise ValueError("Empty inputs for aggregator.")
            tensors = [self._to_BHD(x) for x in h_dict]
        elif isinstance(h_dict, dict):
            ordered = []
            for r in self.rel_names:
                if r in h_dict:
                    ordered.append(self._to_BHD(h_dict[r]))
            if len(ordered) == 0:
                ordered = [self._to_BHD(x) for x in h_dict.values()]
            tensors = ordered
        else:
            raise TypeError(f"Unsupported inputs type for aggregator: {type(h_dict)}")

        if len(tensors) == 1:
            x = tensors[0]  # [B,H,D]
            return x.reshape(x.size(0), self.out_dim)  # [B, H*D]

        # Multi-relation: Attention over relation dimension per head
        T = torch.stack(tensors, dim=0).permute(1, 2, 0, 3).contiguous()
        B, H, R, D = T.shape
        assert H == self.num_heads and D == self.head_dim

        scores = self.score_mlp(T).squeeze(-1)  # [B,H,R]
        alpha = torch.softmax(scores, dim=2)
        alpha = self.dropout(alpha)

        out_BHD = (alpha.unsqueeze(-1) * T).sum(dim=2)  # [B,H,D]
        return out_BHD.reshape(B, self.out_dim)  # [B, H*D]


# HeteroGNN with Layer-wise Regularization
class HeteroGNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, num_layers, num_heads=8,
                 reg_ntypes=None, use_attention=True):
        super().__init__()

        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.num_heads = num_heads if use_attention else 1
        self.use_attention = use_attention
        self.num_heads_in = num_heads if use_attention else 1
        self.num_heads_hidden = num_heads if use_attention else 1
        self.num_heads_out = num_heads if use_attention else 1
        self.drop_rate = 0.2
        self.reg_ntypes = reg_ntypes if reg_ntypes else set()

        if use_attention:
            # Logic when using attention mechanism
            assert hid_feats % num_heads == 0, f"Hidden dimension {hid_feats} must be divisible by head count {num_heads}"
            assert out_feats % num_heads == 0, f"Output dimension {out_feats} must be divisible by head count {num_heads}"

        # Input Layer
        if self.num_layers == 1:
            if use_attention:
                self.agg_in = RelationAttentionAggregator(rel_names, out_feats, num_heads=self.num_heads_in,
                                                          attn_drop=0.1, reg_ntypes=reg_ntypes)
                self.conv_in = dglnn.HeteroGraphConv(
                    {etype: dglnn.GATv2Conv(in_feats, out_feats // self.num_heads_in, self.num_heads_in,
                                            attn_drop=self.drop_rate)
                     for etype in rel_names},
                    aggregate=self.agg_in
                )
            else:
                # Standard GNN: Use GraphConv
                self.agg_in = SimpleAggregator(rel_names)
                self.conv_in = dglnn.HeteroGraphConv(
                    {etype: dglnn.GraphConv(in_feats, out_feats, bias=True)
                     for etype in rel_names},
                    aggregate=self.agg_in
                )

            self.bn = nn.BatchNorm1d(out_feats)
            self.bn1 = nn.BatchNorm1d(out_feats)
            self.bn2 = nn.BatchNorm1d(out_feats)

        else:
            if use_attention:
                self.agg_in = RelationAttentionAggregator(rel_names, hid_feats, num_heads=self.num_heads_in,
                                                          attn_drop=0.1, reg_ntypes=reg_ntypes)
                self.conv_in = dglnn.HeteroGraphConv(
                    {etype: dglnn.GATv2Conv(in_feats, hid_feats // self.num_heads_in, self.num_heads_in,
                                            attn_drop=self.drop_rate)
                     for etype in rel_names},
                    aggregate=self.agg_in
                )
            else:
                # Standard GNN
                self.agg_in = SimpleAggregator(rel_names)
                self.conv_in = dglnn.HeteroGraphConv(
                    {etype: dglnn.GraphConv(in_feats, hid_feats, bias=True)
                     for etype in rel_names},
                    aggregate=self.agg_in
                )

            # Hidden layers
            self.agg_hiddens = nn.ModuleList()
            self.h_layers = nn.ModuleList()

            for l in range(self.num_layers - 2):
                if use_attention:
                    agg = RelationAttentionAggregator(rel_names, self.hid_feats, num_heads=self.num_heads_hidden,
                                                      attn_drop=0.1, reg_ntypes=reg_ntypes)
                    conv = dglnn.HeteroGraphConv(
                        {etype: dglnn.GATv2Conv(self.hid_feats, self.hid_feats // self.num_heads_hidden,
                                                self.num_heads_hidden, attn_drop=self.drop_rate)
                         for etype in rel_names},
                        aggregate=agg
                    )
                else:
                    agg = SimpleAggregator(rel_names)
                    conv = dglnn.HeteroGraphConv(
                        {etype: dglnn.GraphConv(self.hid_feats, self.hid_feats, bias=True)
                         for etype in rel_names},
                        aggregate=agg
                    )

                self.agg_hiddens.append(agg)
                self.h_layers.append(conv)

            # Output layer
            if use_attention:
                self.agg_out = RelationAttentionAggregator(rel_names, self.out_feats, num_heads=self.num_heads_out,
                                                           attn_drop=0.1, reg_ntypes=reg_ntypes)
                self.conv_out = dglnn.HeteroGraphConv(
                    {etype: dglnn.GATv2Conv(self.hid_feats, self.out_feats // self.num_heads_out, self.num_heads_out)
                     for etype in rel_names},
                    aggregate=self.agg_out
                )
            else:
                self.agg_out = SimpleAggregator(rel_names)
                self.conv_out = dglnn.HeteroGraphConv(
                    {etype: dglnn.GraphConv(self.hid_feats, self.out_feats, bias=True)
                     for etype in rel_names},
                    aggregate=self.agg_out
                )

            # BN
            self.bns = nn.ModuleList([nn.BatchNorm1d(self.hid_feats) for _ in range(self.num_layers + 1)])
            self.bns1 = nn.ModuleList([nn.BatchNorm1d(self.hid_feats) for _ in range(self.num_layers + 1)])
            self.bns2 = nn.ModuleList([nn.BatchNorm1d(self.hid_feats) for _ in range(self.num_layers + 1)])

    def forward(self, blocks, inputs):
        """Standard forward, no regularization calculation"""
        h = self.conv_in(blocks[0], inputs)
        self.rel_list = list(h.keys())

        if self.num_layers == 1:
            for rel in self.rel_list:
                if rel == 'ingredient' and rel in h and h[rel].size(0) > 0:
                    h[rel] = F.leaky_relu(self.bn(h[rel].view(-1, self.out_feats)))
                elif rel == 'target' and rel in h and h[rel].size(0) > 0:
                    h[rel] = F.leaky_relu(self.bn1(h[rel].view(-1, self.out_feats)))
                elif rel == 'herb' and rel in h and h[rel].size(0) > 0:
                    h[rel] = F.leaky_relu(self.bn2(h[rel].view(-1, self.out_feats)))
        else:
            if 'ingredient' in h and h['ingredient'].size(0) > 0:
                h['ingredient'] = F.leaky_relu(self.bns[0](h['ingredient'].view(-1, self.hid_feats)))
            if 'target' in h and h['target'].size(0) > 0:
                h['target'] = F.leaky_relu(self.bns1[0](h['target'].view(-1, self.hid_feats)))
            if 'herb' in h and h['herb'].size(0) > 0:
                h['herb'] = F.leaky_relu(self.bns2[0](h['herb'].view(-1, self.hid_feats)))

            for l in range(self.num_layers - 2):
                h = self.h_layers[l](blocks[l + 1], h)
                if 'ingredient' in h and h['ingredient'].size(0) > 0:
                    h['ingredient'] = F.leaky_relu(self.bns[l + 1](h['ingredient'].view(-1, self.hid_feats)))
                if 'target' in h and h['target'].size(0) > 0:
                    h['target'] = F.leaky_relu(self.bns1[l + 1](h['target'].view(-1, self.hid_feats)))
                if 'herb' in h and h['herb'].size(0) > 0:
                    h['herb'] = F.leaky_relu(self.bns2[l + 1](h['herb'].view(-1, self.hid_feats)))

            h = self.conv_out(blocks[self.num_layers - 1], h)

        h = {k: v.view(-1, self.out_feats) if v.size(0) > 0 else v for k, v in h.items()}
        return h

    def forward_with_headwise(self, blocks, inputs, head_mask=None):
        """
        Modified forward propagation, supports layer-wise masking.
        head_mask: Can be Tensor (global mask) or Dict {layer_idx: Tensor} (layer-specific mask)
        """
        # If attention is not used, return standard forward results
        if not self.use_attention:
            h = self.forward(blocks, inputs)
            headwise = {k: v.unsqueeze(1) for k, v in h.items()}
            return h, headwise, []

        # Define internal helper function to apply mask
        def apply_mask_to_h(h_dict, layer_idx, current_num_heads):
            if head_mask is None:
                return h_dict

            # 1. Determine mask for current layer
            mask = None
            if isinstance(head_mask, dict):
                # If dict, apply only if key matches layer_idx
                if layer_idx in head_mask:
                    mask = head_mask[layer_idx]
            elif torch.is_tensor(head_mask):
                # If Tensor, apply globally (backward compatibility)
                mask = head_mask

            if mask is None:
                return h_dict

            # 2. Apply mask
            # Tensor shape in h_dict is usually [N, H*D] (flattened by aggregator)
            # Need to reshape -> mask -> flatten
            masked_h = {}
            # Ensure mask is on correct device
            device = next(iter(h_dict.values())).device
            mask = mask.to(device).view(1, current_num_heads, 1)

            for ntype, tensor in h_dict.items():
                if tensor.numel() > 0:
                    N, Dim = tensor.shape
                    # Reshape to restore head dimension
                    tensor_reshaped = tensor.view(N, current_num_heads, -1)
                    # Multiplicative masking
                    tensor_masked = tensor_reshaped * mask
                    # Flatten back to [N, H*D] for next layer
                    masked_h[ntype] = tensor_masked.view(N, Dim)
                else:
                    masked_h[ntype] = tensor
            return masked_h

        layer_headwise_list = []  # Store headwise output for each layer

        # Layer 0 (Input Layer)
        h = self.conv_in(blocks[0], inputs)

        # 1. Collect headwise for regularization (before Mask)
        layer_headwise = {}
        for ntype, tensor in h.items():
            if ntype in self.reg_ntypes and tensor.size(0) > 0:
                # Determine head count based on layer number
                num_h = self.num_heads_out if self.num_layers == 1 else self.num_heads_in
                layer_headwise[ntype] = tensor.view(tensor.size(0), num_h, -1)
        if layer_headwise:
            layer_headwise_list.append(layer_headwise)

        # 2. Apply Mask immediately before data flows to next layer
        # Layer Index = 0
        current_heads = self.num_heads_out if self.num_layers == 1 else self.num_heads_in
        h = apply_mask_to_h(h, 0, current_heads)

        #  Single Layer Case
        if self.num_layers == 1:
            # Activation and BN
            if 'ingredient' in h and h['ingredient'].size(0) > 0:
                h['ingredient'] = F.leaky_relu(self.bn(h['ingredient'].view(-1, self.out_feats)))
            if 'target' in h and h['target'].size(0) > 0:
                h['target'] = F.leaky_relu(self.bn1(h['target'].view(-1, self.out_feats)))
            if 'herb' in h and h['herb'].size(0) > 0:
                h['herb'] = F.leaky_relu(self.bn2(h['herb'].view(-1, self.out_feats)))

            # Prepare final output
            headwise = {}
            flat = {}
            for ntype, tensor in h.items():
                N = tensor.size(0)
                if N > 0:
                    flat[ntype] = tensor.view(-1, self.out_feats)
                    if tensor.size(-1) == self.out_feats:
                        headwise[ntype] = tensor.view(N, self.num_heads_out, self.out_feats // self.num_heads_out)
                    else:
                        headwise[ntype] = tensor
                else:
                    flat[ntype] = tensor
                    headwise[ntype] = tensor

            return flat, headwise, layer_headwise_list

        # Multi-Layer Case
        else:
            # Input Layer BN & Activation
            if 'ingredient' in h:
                h['ingredient'] = F.leaky_relu(self.bns[0](h['ingredient'].view(-1, self.hid_feats)))
            if 'target' in h:
                h['target'] = F.leaky_relu(self.bns1[0](h['target'].view(-1, self.hid_feats)))
            if 'herb' in h:
                h['herb'] = F.leaky_relu(self.bns2[0](h['herb'].view(-1, self.hid_feats)))

            # Hidden layer loop
            for l in range(self.num_layers - 2):
                # GNN Propagation
                h = self.h_layers[l](blocks[l + 1], h)

                # 1. Collect regularization info
                layer_headwise = {}
                for ntype, tensor in h.items():
                    if ntype in self.reg_ntypes and tensor.size(0) > 0:
                        layer_headwise[ntype] = tensor.view(tensor.size(0), self.num_heads_hidden, -1)
                if layer_headwise:
                    layer_headwise_list.append(layer_headwise)

                # 2. Apply Mask (Layer Index = l + 1)
                h = apply_mask_to_h(h, l + 1, self.num_heads_hidden)

                # BN & Activation
                if 'ingredient' in h:
                    h['ingredient'] = F.leaky_relu(self.bns[l + 1](h['ingredient'].view(-1, self.hid_feats)))
                if 'target' in h:
                    h['target'] = F.leaky_relu(self.bns1[l + 1](h['target'].view(-1, self.hid_feats)))
                if 'herb' in h:
                    h['herb'] = F.leaky_relu(self.bns2[l + 1](h['herb'].view(-1, self.hid_feats)))

            # Output layer
            hout = self.conv_out(blocks[self.num_layers - 1], h)

            # 1. Collect regularization info (Output Layer)
            layer_headwise = {}
            for ntype, tensor in hout.items():
                if ntype in self.reg_ntypes and tensor.size(0) > 0:
                    layer_headwise[ntype] = tensor.view(tensor.size(0), self.num_heads_out, -1)
            if layer_headwise:
                layer_headwise_list.append(layer_headwise)

            # 2. Apply Mask (Layer Index = num_layers - 1)
            hout = apply_mask_to_h(hout, self.num_layers - 1, self.num_heads_out)

            # Prepare final output
            headwise = {}
            flat = {}

            for ntype, tensor in hout.items():
                N = tensor.size(0)
                if N > 0:
                    if tensor.dim() == 2:
                        # Reshape to [N, H, D_head] to return headwise
                        headwise_tensor = tensor.view(N, self.num_heads_out, self.out_feats // self.num_heads_out)
                    else:
                        headwise_tensor = tensor

                    headwise[ntype] = headwise_tensor
                    # flat uses masked tensor directly (already [N, H*D])
                    flat[ntype] = tensor.reshape(-1, self.out_feats)
                else:
                    headwise[ntype] = tensor
                    flat[ntype] = tensor

            return flat, headwise, layer_headwise_list


#  Predictor Classes
class MLPScorePredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dims=None, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, edge_subgraph, h):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['h'] = h
            scores = {}
            for etype in edge_subgraph.canonical_etypes:
                src_type, _, dst_type = etype
                src, dst = edge_subgraph.edges(etype=etype)
                src_embeddings = h[src_type][src]
                dst_embeddings = h[dst_type][dst]
                edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
                edge_scores = self.mlp(edge_embeddings).squeeze(-1)
                scores[etype] = edge_scores
            return scores


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, h):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['h'] = h
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return edge_subgraph.edata['score']


class MixedPredictor(nn.Module):
    """
    Mixed Predictor: Combines advantages of Dot Product and MLP
    Supports fusion strategies: residual, weighted, gated
    """

    def __init__(self, embedding_dim, hidden_dims=None, dropout=0.2,
                 fusion_method='residual', learnable_weight=False, residual_scale=1.0):
        super().__init__()

        # MLP predictor branch
        self.mlp_predictor = MLPScorePredictor(embedding_dim, hidden_dims, dropout)

        # Fusion method
        self.fusion_method = fusion_method
        self.residual_scale = residual_scale

        # Learnable fusion weight
        if learnable_weight and fusion_method == 'weighted':
            # Initialize to 0.5, ensure [0,1] range via sigmoid
            self.alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        else:
            self.alpha = None

        # Gated fusion mechanism
        if fusion_method == 'gated':
            # Learn a gating network for each edge type
            self.gate_net = nn.Sequential(
                nn.Linear(embedding_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 2),  # Output 2 gate values: [dot_gate, mlp_gate]
                nn.Softmax(dim=-1)
            )

    def _compute_dot_product(self, edge_subgraph, h):
        """Compute dot product scores"""
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['h'] = h
            scores = {}
            for etype in edge_subgraph.canonical_etypes:
                src_type, _, dst_type = etype
                src, dst = edge_subgraph.edges(etype=etype)
                src_embeddings = h[src_type][src]
                dst_embeddings = h[dst_type][dst]
                # Compute dot product
                dot_scores = (src_embeddings * dst_embeddings).sum(dim=1)
                scores[etype] = dot_scores
            return scores

    def forward(self, edge_subgraph, h):
        # 1. Compute dot product scores (base scores)
        dot_scores = self._compute_dot_product(edge_subgraph, h)

        # 2. Compute MLP scores (non-linear correction)
        mlp_scores = self.mlp_predictor(edge_subgraph, h)

        # 3. Combine scores based on fusion strategy
        final_scores = {}

        if self.fusion_method == 'residual':
            # Residual sum: Dot product as base, MLP as residual correction
            # Final = Dot + scale * MLP
            for etype in edge_subgraph.canonical_etypes:
                final_scores[etype] = dot_scores[etype] + self.residual_scale * mlp_scores[etype]

        elif self.fusion_method == 'weighted':
            # Weighted average: Use fixed or learnable weights
            if self.alpha is not None:
                # Learnable weight
                alpha = torch.sigmoid(self.alpha)
            else:
                # Fixed weight 0.5
                alpha = 0.5

            for etype in edge_subgraph.canonical_etypes:
                final_scores[etype] = alpha * dot_scores[etype] + (1 - alpha) * mlp_scores[etype]

        elif self.fusion_method == 'gated':
            # Gated fusion: Dynamically compute fusion weights for each edge
            for etype in edge_subgraph.canonical_etypes:
                src_type, _, dst_type = etype
                src, dst = edge_subgraph.edges(etype=etype)
                src_embeddings = h[src_type][src]
                dst_embeddings = h[dst_type][dst]

                # Concatenate source and target embeddings
                edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)

                # Compute gating weights
                gates = self.gate_net(edge_embeddings)  # [num_edges, 2]
                dot_gate = gates[:, 0]
                mlp_gate = gates[:, 1]

                # Weighted combination
                final_scores[etype] = dot_gate * dot_scores[etype] + mlp_gate * mlp_scores[etype]

        return final_scores

    def get_component_scores(self, edge_subgraph, h):
        """Get component scores for analysis and debugging"""
        dot_scores = self._compute_dot_product(edge_subgraph, h)
        mlp_scores = self.mlp_predictor(edge_subgraph, h)
        return dot_scores, mlp_scores


def feature_mapping_mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, out_dim)
    )


def feature_mapping_mlp_with_norm(in_dim, out_dim):
    """Version with LayerNorm, used for ingredient and target features"""
    return nn.Sequential(
        nn.LayerNorm(in_dim),  # Add LayerNorm as the first layer
        nn.Linear(in_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, out_dim)
    )


# Model Class
class Model(nn.Module):
    def __init__(self, args, etypes, predictor_type='md'):
        super().__init__()

        # Feature mapping layers (select encoder based on dataset and args)
        # Check if Zero-aware encoder is used (only for herb nodes in herb dataset)
        if (args.dataset == 'herb' and
                not args.remove_herb and
                getattr(args, 'use_herb_zero_aware', False)):
            # Use hybrid herb encoder
            self.mapping_herb = HerbEncoder(
                input_dim=args.herb_in_dim,
                output_dim=args.in_dim,
                dropout=getattr(args, 'herb_encoder_dropout', 0.1),
                zero_importance_init=getattr(args, 'zero_importance_init', 0.5)
            )
        else:
            # Use standard MLP
            self.mapping_herb = feature_mapping_mlp(args.herb_in_dim, args.in_dim)

        # ingredient and target use MLP with LayerNorm
        self.mapping_ingredient = feature_mapping_mlp_with_norm(args.ingredient_in_dim, args.in_dim)
        self.mapping_target = feature_mapping_mlp_with_norm(args.target_in_dim, args.in_dim)

        # Determine which GNN to use
        self.use_simple_gnn = getattr(args, 'use_simple_gnn', False)

        if self.use_simple_gnn:
            # Use simple GNN
            self.HeteroGNN = SimpleGNN(args.in_dim, args.h_dim, args.out_dim, args.num_layers)
            use_attention = False
        else:
            # Use HeteroGNN
            use_attention = getattr(args, 'use_attention', True)
            self.HeteroGNN = HeteroGNN(args.in_dim, args.h_dim, args.out_dim,
                                       etypes, args.num_layers, args.num_heads,
                                       reg_ntypes=set(args.reg_ntypes),
                                       use_attention=use_attention)

        self.predictor_type = predictor_type

        # Initialize corresponding predictor based on type
        if predictor_type == 'mlp':
            # Use MLP predictor only
            hidden_dims = args.mlp_hidden_dims if hasattr(args, 'mlp_hidden_dims') else [128, 64, 32]
            dropout = args.mlp_dropout if hasattr(args, 'mlp_dropout') else 0.2
            self.pred = MLPScorePredictor(args.out_dim, hidden_dims, dropout)

        elif predictor_type == 'dot':
            # Use Dot Product predictor only
            self.pred = ScorePredictor()

        elif predictor_type == 'md':
            # Use Mixed predictor
            hidden_dims = args.mlp_hidden_dims if hasattr(args, 'mlp_hidden_dims') else [128, 64, 32]
            dropout = args.mlp_dropout if hasattr(args, 'mlp_dropout') else 0.2
            fusion_method = getattr(args, 'fusion_method', 'residual')
            learnable_weight = getattr(args, 'learnable_weight', False)
            residual_scale = getattr(args, 'residual_scale', 1.0)

            self.pred = MixedPredictor(
                args.out_dim,
                hidden_dims,
                dropout,
                fusion_method=fusion_method,
                learnable_weight=learnable_weight,
                residual_scale=residual_scale
            )
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")

        # Regularization parameters (valid only when using attention)
        self.use_attention = use_attention and not self.use_simple_gnn
        if self.use_attention:
            self.lambda_head_div = getattr(args, 'lambda_head_div', 0.0)
            self.lambda_head_entropy = getattr(args, 'lambda_head_entropy', 0.0)
            self.reg_ntypes = set(getattr(args, 'reg_ntypes', ['ingredient', 'target']))
            self.p_headdrop = getattr(args, 'p_headdrop', 0.01)
        else:
            self.lambda_head_div = 0.0
            self.lambda_head_entropy = 0.0
            self.p_headdrop = 0.0

        self.layer_reg_decay = getattr(args, 'layer_reg_decay', 0.8)

    def forward(self, args, positive_graph, negative_graph, blocks, x):
        """
        Accumulate regularization loss for each layer
        """
        # 1) Feature mapping (always use pretrained embeddings)
        mapped_features = {}
        if 'herb' in x:
            mapped_features['herb'] = self.mapping_herb(x['herb'])
        if 'ingredient' in x:
            mapped_features['ingredient'] = self.mapping_ingredient(x['ingredient'])
        if 'target' in x:
            mapped_features['target'] = self.mapping_target(x['target'])

        # 2) HeadDrop (only when using attention)
        head_mask = None
        if self.training and self.use_attention and self.p_headdrop > 0.0:
            H = self.HeteroGNN.num_heads if hasattr(self.HeteroGNN, 'num_heads') else 1
            keep = torch.bernoulli(torch.full((H,), 1.0 - self.p_headdrop,
                                              device=next(self.parameters()).device))
            if keep.sum() == 0:
                keep[torch.randint(0, H, (1,), device=keep.device)] = 1.0
            head_mask = keep

        # 3) Get flat, headwise, and layer-wise headwise outputs
        flat_embeddings, headwise_embeddings, layer_headwise_list = self.HeteroGNN.forward_with_headwise(
            blocks, mapped_features, head_mask=head_mask
        )

        # 4) Scoring
        pos_score = self.pred(positive_graph, flat_embeddings)
        neg_score = self.pred(negative_graph, flat_embeddings)

        # 5) Compute regularization (only when using attention)
        reg_terms = {'div': torch.tensor(0.0, device=next(self.parameters()).device),
                     'ent': torch.tensor(0.0, device=next(self.parameters()).device)}

        if self.training and self.use_attention and (self.lambda_head_div > 0.0 or self.lambda_head_entropy > 0.0):
            total_div = torch.tensor(0.0, device=next(self.parameters()).device)
            total_ent = torch.tensor(0.0, device=next(self.parameters()).device)

            num_layers = len(layer_headwise_list)
            if num_layers > 0:
                for layer_idx, layer_headwise in enumerate(layer_headwise_list):
                    layer_div, layer_ent = self._head_regularizers(layer_headwise)
                    layer_weight = self.layer_reg_decay ** layer_idx
                    total_div = total_div + layer_weight * layer_div
                    total_ent = total_ent + layer_weight * layer_ent

                weight_sum = sum(self.layer_reg_decay ** i for i in range(num_layers))
                reg_terms['div'] = total_div / weight_sum if weight_sum > 0 else total_div
                reg_terms['ent'] = total_ent / weight_sum if weight_sum > 0 else total_ent

        return pos_score, neg_score, reg_terms

    def encode_with_heads(self, args, blocks, x, head_mask=None):
        """Interface for head_analysis.py"""
        mapped_features = {}
        if 'herb' in x:
            mapped_features['herb'] = self.mapping_herb(x['herb'])
        if 'ingredient' in x:
            mapped_features['ingredient'] = self.mapping_ingredient(x['ingredient'])
        if 'target' in x:
            mapped_features['target'] = self.mapping_target(x['target'])

        flat, headwise, _ = self.HeteroGNN.forward_with_headwise(blocks, mapped_features, head_mask=head_mask)
        return flat, headwise

    def _head_regularizers(self, headwise_dict):
        """Compute head regularization loss"""
        device = next(self.parameters()).device
        div_total = torch.zeros((), device=device)
        ent_total = torch.zeros((), device=device)

        for ntype, tens in headwise_dict.items():
            if ntype not in self.reg_ntypes:
                continue

            # tens: (N, H, d)
            if tens.dim() != 3:
                continue

            N, H, d = tens.shape
            if N == 0 or H <= 1:
                continue

            # Vector norm for each node and head
            norms = torch.linalg.vector_norm(tens, dim=2) + 1e-12  # (N, H)

            # [1] Decorrelation: Correlation matrix over dimension H
            z = (norms - norms.mean(dim=0, keepdim=True))
            std = norms.std(dim=0, keepdim=True) + 1e-12
            z = z / std
            corr = (z.t() @ z) / max(N - 1, 1)  # (H, H)
            off_diag = corr - torch.diag(torch.diag(corr))
            div_term = (off_diag ** 2).mean()
            div_total = div_total + div_term

            # [2] Entropy: Mean entropy of head distribution for each node
            p = norms / norms.sum(dim=1, keepdim=True)
            entropy = -(p * (p.clamp_min(1e-12).log())).sum(dim=1).mean()
            ent_total = ent_total - entropy  # Negative sign: Maximize entropy

        return div_total, ent_total