import torch
import torch.nn as nn

from spurs.models.stability.org_transfer_model import DecLayer, cat_neighbors_nodes, gather_nodes


class LigandMPNNEncoder(nn.Module):
    """LigandMPNN-style structural encoder without autoregressive decoding.

    Inputs combine protein nodes, ligand nodes and protein-ligand graph edges.
    Decoder attention is full-visibility (all ones), while graph masking is still
    controlled by node/edge masks and k-NN adjacency (``knn_idx``).
    """

    def __init__(
        self,
        protein_node_dim: int,
        ligand_node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        wt_aa_vocab_size: int = 21,
        wt_aa_embed_dim: int = 128,
        num_decoder_layers: int = 3,
        dropout: float = 0.1,
        kv_dim: int = 1280,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.protein_in = nn.Linear(protein_node_dim, hidden_dim)
        self.ligand_in = nn.Linear(ligand_node_dim, hidden_dim)
        self.edge_in = nn.Linear(edge_dim, hidden_dim)

        self.decoder_layers = nn.ModuleList(
            [DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout) for _ in range(num_decoder_layers)]
        )

        self.wt_aa_embedding = nn.Embedding(wt_aa_vocab_size, wt_aa_embed_dim)
        self.kv_proj = nn.Linear(hidden_dim + wt_aa_embed_dim, kv_dim)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        protein_nodes: torch.Tensor,
        ligand_nodes: torch.Tensor,
        protein_ligand_edges: torch.Tensor,
        knn_idx: torch.Tensor,
        wt_residue_idx: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ):
        """Return geometric features for deepest adapter K/V.

        Args:
            protein_nodes: [B, Np, Dp]
            ligand_nodes: [B, Nl, Dl]
            protein_ligand_edges: [B, Np+Nl, K, De]
            knn_idx: [B, Np+Nl, K] neighbor indices over concatenated nodes
            wt_residue_idx: [B, Np] WT amino-acid token ids
            protein_mask: [B, Np]
            ligand_mask: [B, Nl]
            edge_mask: [B, Np+Nl, K]

        Returns:
            dict with:
                V_dec: [B, Np, hidden_dim]
                E_AA: [B, Np, wt_aa_embed_dim]
                F_geo: [B, Np, hidden_dim + wt_aa_embed_dim]
                F_geo_proj: [B, Np, 1280]
        """
        n_protein = protein_nodes.size(1)

        h_protein = self.protein_in(protein_nodes)
        h_ligand = self.ligand_in(ligand_nodes)
        h_V = torch.cat([h_protein, h_ligand], dim=1)
        h_E = self.edge_in(protein_ligand_edges)

        node_mask = torch.cat([protein_mask, ligand_mask], dim=1)

        # Full-visibility attention mask (non-autoregressive), still constrained by
        # graph mask and k-NN adjacency.
        full_visibility = torch.ones_like(edge_mask)
        if node_mask is not None:
            knn_valid = gather_nodes(node_mask.unsqueeze(-1), knn_idx).squeeze(-1)
            full_visibility = full_visibility * knn_valid * node_mask.unsqueeze(-1)
        if edge_mask is not None:
            full_visibility = full_visibility * edge_mask

        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_E, knn_idx)
            h_ESV = h_ESV * full_visibility.unsqueeze(-1)
            h_V = layer(h_V, h_ESV, mask_V=node_mask, mask_attend=full_visibility)

        V_dec = h_V[:, :n_protein, :]
        E_AA = self.wt_aa_embedding(wt_residue_idx)
        F_geo = torch.cat([V_dec, E_AA], dim=-1)
        F_geo_proj = self.kv_proj(F_geo)

        return {
            "V_dec": V_dec,
            "E_AA": E_AA,
            "F_geo": F_geo,
            "F_geo_proj": F_geo_proj,
        }
