import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def index_points(points, idx):
    """
    Gather each point's K neighbor points.

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, K]
    Return:
        new_points:, indexed points data, [B, S, K, C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()

        self.point_net = nn.Sequential(
                nn.Linear(3, d_model // 2),
                nn.BatchNorm1d(d_model // 2),
                nn.ReLU(True),

                nn.Linear(d_model // 2, d_model),
                nn.BatchNorm1d(d_model),
                nn.ReLU(True)
            )

        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

    def forward(self, xyz, features):
        """
        Computing feature for each point using transformer.

        Args:
            xyz: [b, n, 3]
            features: [b, n, c]

        Returns:
            features: [b, n, c]
        """
        b, n, _ = xyz.shape
        xyz = torch.reshape(xyz, (b*n, -1))
        pos_enc = self.point_net(xyz)  # [b* n, c]
        pos_enc = torch.reshape(pos_enc, (b, n, -1))  # [b, n, c]
        new_features = pos_enc + features  # [b, n, c]

        q = self.w_qs(new_features)  # [b, n, c]
        k = self.w_ks(new_features).permute(0, 2, 1)  # [b, c, n]
        v = self.w_vs(new_features)  # [b, n, c]

        weights = q @ k  # [b, n, n]
        weights = weights / np.sqrt(q.size(-1))
        weights = F.softmax(weights, dim=-1)  # [b, n, n]

        grouped_features = weights @ v  # [b, n, c]
        return grouped_features

class Mutual_TransformerBlock(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()

        self.pc0_q = nn.Linear(d_model, d_model, bias=False)
        self.pc0_k = nn.Linear(d_model, d_model, bias=False)
        self.pc0_v = nn.Linear(d_model, d_model, bias=False)

        self.pc1_q = nn.Linear(d_model, d_model, bias=False)
        self.pc1_k = nn.Linear(d_model, d_model, bias=False)
        self.pc1_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, pc0_fea, pc1_fea):
        """
        Computing feature for each point using transformer.

        Args:
            pc0_fea: [b, n, c]
            pc1_fea: [b, n, c]

        Returns:
            pc0_aware_features: [b, n, c]
            pc1_aware_features: [b, n, c]
        """

        pc0_q = self.pc0_q(pc0_fea)  # [b, n, c]
        pc0_k = self.pc0_k(pc0_fea).permute(0, 2, 1)  # [b, c, n]
        pc0_v = self.pc0_v(pc0_fea)

        pc1_q = self.pc1_q(pc1_fea) # [b, n, c]
        pc1_k = self.pc1_k(pc1_fea).permute(0, 2, 1) # [b, c, n]
        pc1_v = self.pc1_v(pc1_fea) # [b, n, c]

        pc0_weights = (pc1_q @ pc0_k) / np.sqrt(pc1_q.size(-1)) # [b, n, n]
        pc0_weights = F.softmax(pc0_weights, dim=-1) # [b, n, n]
        pc0_aware_features = pc0_weights @ pc0_v # [b, n, c]

        pc1_weights = (pc0_q @ pc1_k) / np.sqrt(pc0_q.size(-1)) # [b, n, n]
        pc1_weights = F.softmax(pc1_weights, dim=-1) # [b, n, n]
        pc1_aware_features = pc1_weights @ pc1_v # [b, n, c]

        return pc0_aware_features, pc1_aware_features