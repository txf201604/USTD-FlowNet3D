import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from FPT_utils.models.transformer_base import LocalSelfAttentionBase, ResidualBlockWithPointsBase
from FPT_utils.models.common import stride_centroids, downsample_points, downsample_embeddings
import FPT_utils.cuda_ops.functions.sparse_ops as ops


class MaxPoolWithPoints(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        assert kernel_size == 2 and stride == 2
        super(MaxPoolWithPoints, self).__init__()
        self.pool = ME.MinkowskiMaxPooling(kernel_size=kernel_size, stride=stride, dimension=3)

    def forward(self, stensor, points, counts):
        assert isinstance(stensor, ME.SparseTensor)
        assert len(stensor) == len(points)
        cm = stensor.coordinate_manager
        down_stensor = self.pool(stensor)
        cols, rows = cm.stride_map(stensor.coordinate_map_key, down_stensor.coordinate_map_key)
        size = torch.Size([len(down_stensor), len(stensor)])
        down_points, down_counts = stride_centroids(points, counts, rows, cols, size)
        return down_stensor, down_points, down_counts


####################################
# Layers
####################################
class LightweightSelfAttentionLayer(LocalSelfAttentionBase):
    def __init__(
            self,
            in_channels,
            out_channels=None,
            kernel_size=3,
            stride=1,
            dilation=1,
            num_heads=8,
    ):
        out_channels = in_channels if out_channels is None else out_channels
        assert out_channels % num_heads == 0
        assert kernel_size % 2 == 1
        assert stride == 1, "Currently, this layer only supports stride == 1"
        assert dilation == 1, "Currently, this layer only supports dilation == 1"
        super(LightweightSelfAttentionLayer, self).__init__(kernel_size, stride, dilation, dimension=3)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.num_heads = num_heads
        self.attn_channels = out_channels // num_heads

        self.to_query = nn.Sequential(
            ME.MinkowskiLinear(in_channels, out_channels),
            ME.MinkowskiToFeature()
        )
        self.to_value = nn.Sequential(
            ME.MinkowskiLinear(in_channels, out_channels),
            ME.MinkowskiToFeature()
        )
        self.to_out = nn.Linear(out_channels, out_channels)

        self.inter_pos_enc = nn.Parameter(torch.FloatTensor(self.kernel_volume, self.num_heads, self.attn_channels))
        self.intra_pos_mlp = nn.Sequential(
            nn.Linear(3, 3, bias=False),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        )
        nn.init.normal_(self.inter_pos_enc, 0, 1)

    def forward(self, stensor, norm_points):
        dtype = stensor._F.dtype
        device = stensor._F.device

        # query, key, value, and relative positional encoding
        intra_pos_enc = self.intra_pos_mlp(norm_points)
        stensor = stensor + intra_pos_enc
        q = self.to_query(stensor).view(-1, self.num_heads, self.attn_channels).contiguous()
        v = self.to_value(stensor).view(-1, self.num_heads, self.attn_channels).contiguous()

        # key-query map
        kernel_map, out_key = self.get_kernel_map_and_out_key(stensor)
        kq_map = self.key_query_map_from_kernel_map(kernel_map)

        # attention weights with cosine similarity
        attn = torch.zeros((kq_map.shape[1], self.num_heads), dtype=dtype, device=device)
        norm_q = F.normalize(q, p=2, dim=-1)
        norm_pos_enc = F.normalize(self.inter_pos_enc, p=2, dim=-1)
        attn = ops.dot_product_cuda(norm_q, norm_pos_enc, attn, kq_map)

        # aggregation & the output
        out_F = torch.zeros((len(q), self.num_heads, self.attn_channels),
                            dtype=dtype,
                            device=device)
        kq_indices = self.key_query_indices_from_key_query_map(kq_map)
        out_F = ops.scalar_attention_cuda(attn, v, out_F, kq_indices)
        out_F = self.to_out(out_F.view(-1, self.out_channels).contiguous())
        return ME.SparseTensor(out_F,
                               coordinate_map_key=out_key,
                               coordinate_manager=stensor.coordinate_manager)


####################################
# Blocks
####################################
class LightweightSelfAttentionBlock(ResidualBlockWithPointsBase):
    LAYER = LightweightSelfAttentionLayer


####################################
# Models
####################################
class FastPointTransformer(nn.Module):
    INIT_DIM = 32
    ENC_DIM = 32
    PLANES = (64, 64, 128, 128, 128, 128, 64, 64)
    QMODE = ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
    LAYER = LightweightSelfAttentionLayer
    BLOCK = LightweightSelfAttentionBlock

    def __init__(self, in_channels, out_channels):
        super(FastPointTransformer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.enc_mlp = nn.Sequential(
            nn.Linear(3, self.ENC_DIM, bias=False),
            nn.BatchNorm1d(self.ENC_DIM),
            nn.Tanh(),
            nn.Linear(self.ENC_DIM, self.ENC_DIM, bias=False),
            nn.BatchNorm1d(self.ENC_DIM),
            nn.Tanh()
        )

        self.attn0p1 = self.LAYER(in_channels + self.ENC_DIM, self.INIT_DIM, kernel_size=5)
        self.bn0 = ME.MinkowskiBatchNorm(self.INIT_DIM)

        self.attn1p1 = self.LAYER(self.INIT_DIM, self.PLANES[0])
        self.bn1 = ME.MinkowskiBatchNorm(self.PLANES[0])
        self.block1 = self.BLOCK(self.PLANES[0])

        self.attn2p2 = self.LAYER(self.PLANES[0], self.PLANES[1])
        self.bn2 = ME.MinkowskiBatchNorm(self.PLANES[1])
        self.block2 = self.BLOCK(self.PLANES[1])

        self.attn3p4 = self.LAYER(self.PLANES[1], self.PLANES[2])
        self.bn3 = ME.MinkowskiBatchNorm(self.PLANES[2])
        self.block3 = self.BLOCK(self.PLANES[2])

        self.attn4p8 = self.LAYER(self.PLANES[2], self.PLANES[3])
        self.bn4 = ME.MinkowskiBatchNorm(self.PLANES[3])
        self.block4 = self.BLOCK(self.PLANES[3])

        self.attn5p8 = self.LAYER(self.PLANES[3] + self.PLANES[3], self.PLANES[4])
        self.bn5 = ME.MinkowskiBatchNorm(self.PLANES[4])
        self.block5 = self.BLOCK(self.PLANES[4])

        self.attn6p4 = self.LAYER(self.PLANES[4] + self.PLANES[2], self.PLANES[5])
        self.bn6 = ME.MinkowskiBatchNorm(self.PLANES[5])
        self.block6 = self.BLOCK(self.PLANES[5])

        self.attn7p2 = self.LAYER(self.PLANES[5] + self.PLANES[1], self.PLANES[6])
        self.bn7 = ME.MinkowskiBatchNorm(self.PLANES[6])
        self.block7 = self.BLOCK(self.PLANES[6])

        self.attn8p1 = self.LAYER(self.PLANES[6] + self.PLANES[0], self.PLANES[7])
        self.bn8 = ME.MinkowskiBatchNorm(self.PLANES[7])
        self.block8 = self.BLOCK(self.PLANES[7])

        self.final = ME.MinkowskiConvolution(
            in_channels=self.PLANES[7],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pool = MaxPoolWithPoints()
        self.pooltr = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3)

    @torch.no_grad()
    def normalize_points(self, points, centroids, tensor_map):
        tensor_map = tensor_map if tensor_map.dtype == torch.int64 else tensor_map.long()
        norm_points = points - centroids[tensor_map]
        return norm_points

    @torch.no_grad()
    def normalize_centroids(self, down_points, coordinates, tensor_stride):
        norm_points = (down_points - coordinates[:, 1:]) / tensor_stride - 0.5
        return norm_points

    def voxelize_with_centroids(self, x):
        cm = x.coordinate_manager
        points = x.C[:, 1:]

        out = x.sparse()
        size = torch.Size([len(out), len(x)])
        tensor_map, field_map = cm.field_to_sparse_map(x.coordinate_key, out.coordinate_key)
        points_p1, count_p1 = downsample_points(points, tensor_map, field_map, size)
        norm_points = self.normalize_points(points, points_p1, tensor_map)

        pos_embs = self.enc_mlp(norm_points)
        down_pos_embs = downsample_embeddings(pos_embs, tensor_map, size, mode="avg")
        out = ME.SparseTensor(torch.cat([out.F, down_pos_embs], dim=1),
                              coordinate_map_key=out.coordinate_key,
                              coordinate_manager=cm)

        norm_points_p1 = self.normalize_centroids(points_p1, out.C, out.tensor_stride[0])
        return out, norm_points_p1, points_p1, count_p1

    def forward(self, x):
        out, norm_points_p1, points_p1, count_p1 = self.voxelize_with_centroids(x)
        out = self.relu(self.bn0(self.attn0p1(out, norm_points_p1)))
        out_p1 = self.relu(self.bn1(self.attn1p1(out, norm_points_p1)))

        out, points_p2, count_p2 = self.pool(out_p1, points_p1, count_p1)
        norm_points_p2 = self.normalize_centroids(points_p2, out.C, out.tensor_stride[0])
        out = self.block1(out, norm_points_p2)
        out_p2 = self.relu(self.bn2(self.attn2p2(out, norm_points_p2)))

        out, points_p4, count_p4 = self.pool(out_p2, points_p2, count_p2)
        norm_points_p4 = self.normalize_centroids(points_p4, out.C, out.tensor_stride[0])
        out = self.block2(out, norm_points_p4)
        out_p4 = self.relu(self.bn3(self.attn3p4(out, norm_points_p4)))

        out, points_p8, count_p8 = self.pool(out_p4, points_p4, count_p4)
        norm_points_p8 = self.normalize_centroids(points_p8, out.C, out.tensor_stride[0])
        out = self.block3(out, norm_points_p8)
        out_p8 = self.relu(self.bn4(self.attn4p8(out, norm_points_p8)))

        out, points_p16 = self.pool(out_p8, points_p8, count_p8)[:2]
        norm_points_p16 = self.normalize_centroids(points_p16, out.C, out.tensor_stride[0])
        out = self.block4(out, norm_points_p16)

        out = self.pooltr(out)
        out = ME.cat(out, out_p8)
        out = self.relu(self.bn5(self.attn5p8(out, norm_points_p8)))
        out = self.block5(out, norm_points_p8)

        out = self.pooltr(out)
        out = ME.cat(out, out_p4)
        out = self.relu(self.bn6(self.attn6p4(out, norm_points_p4)))
        out = self.block6(out, norm_points_p4)

        out = self.pooltr(out)
        out = ME.cat(out, out_p2)
        out = self.relu(self.bn7(self.attn7p2(out, norm_points_p2)))
        out = self.block7(out, norm_points_p2)

        out = self.pooltr(out)
        out = ME.cat(out, out_p1)
        out = self.relu(self.bn8(self.attn8p1(out, norm_points_p1)))
        out = self.block8(out, norm_points_p1)

        out = self.final(out)
        return out


class FastPointTransformerFeatureExtractor(nn.Module):
    def __init__(self, voxel_size, normalize_feature=True):
        super(FastPointTransformerFeatureExtractor, self).__init__()


        self.voxel_size = voxel_size
        self.normalize_feature = normalize_feature

        self.in_channels = 3
        self.out_channels = 64
        # self.enc_dim = 32
        self.feature_extractor = FastPointTransformer(self.in_channels, self.out_channels)

    #     self.mlp = nn.Sequential(
    #         nn.Linear(self.out_channels+self.enc_dim, self.out_channels, bias=False),
    #         nn.BatchNorm1d(self.out_channels),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(self.out_channels, self.out_channels)
    #     )
    #
    # def devoxelize_with_centroids(self, point_fea, h_embs):
    #     feature = torch.cat([point_fea, h_embs], dim=-1)
    #     feature_out = self.mlp(feature)
    #     return feature_out

    def forward(self, st_1, st_2, xyz_1, xyz_2, k_values):
        B1, N1, _ = xyz_1.shape
        B2, N2, _ = xyz_2.shape

        # feature_1, pos_embs_1 = self.feature_extractor(st_1)
        # feature_2, pos_embs_2 = self.feature_extractor(st_2)
        feature_1 = self.feature_extractor(st_1)
        feature_2 = self.feature_extractor(st_2)

        if self.normalize_feature:
            feature_1 = ME.SparseTensor(
                        feature_1.F / torch.norm(feature_1.F, p=2, dim=1, keepdim=True),
                        coordinate_map_key=feature_1.coordinate_map_key,
                        coordinate_manager=feature_1.coordinate_manager)

            feature_2 = ME.SparseTensor(
                        feature_2.F / torch.norm(feature_2.F, p=2, dim=1, keepdim=True),
                        coordinate_map_key=feature_2.coordinate_map_key,
                        coordinate_manager=feature_2.coordinate_manager)

        up_f_1 = self.upsample(xyz_1, feature_1, k_value=k_values)
        up_f_2 = self.upsample(xyz_2, feature_2, k_value=k_values)

        # f_1 = torch.stack(up_f_1, dim=0).view(-1, self.out_channels).contiguous()
        # f_2 = torch.stack(up_f_2, dim=0).view(-1, self.out_channels).contiguous()
        f_1 = torch.stack(up_f_1, dim=0)
        f_2 = torch.stack(up_f_2, dim=0)

        # fea_1 = self.devoxelize_with_centroids(f_1, pos_embs_1).view(B1, N1, self.out_channels).contiguous()
        # fea_2 = self.devoxelize_with_centroids(f_2, pos_embs_2).view(B2, N2, self.out_channels).contiguous()

        return f_1, f_2

    def upsample(self, xyz, sparse_tensor, k_value=3):
        dense_flow = []
        b, n, _ = xyz.shape
        for b_idx in range(b):
            sparse_xyz = sparse_tensor.coordinates_at(b_idx).cuda() * self.voxel_size
            # sparse_xyz = sparse_tensor.coordinates_at(b_idx).cuda()
            sparse_feature = sparse_tensor.features_at(b_idx)

            sqr_dist = self.pairwise_distance(xyz[b_idx], sparse_xyz, normalized=False).squeeze(0)
            sqr_dist, group_idx = torch.topk(sqr_dist, k_value, dim=-1, largest=False, sorted=False)

            dist = torch.sqrt(sqr_dist)
            norm = torch.sum(1 / (dist + 1e-7), dim=1, keepdim=True)
            weight = ((1 / (dist + 1e-7)) / norm).unsqueeze(-1)

            sparse_flow = sparse_feature[group_idx.reshape(-1), :].reshape(n, k_value, -1)
            dense_flow.append(torch.sum(weight * sparse_flow, dim=1))
        return dense_flow

    def pairwise_distance(self, src, dst, normalized=True):
        if len(src.shape) == 2:
            src = src.unsqueeze(0)
            dst = dst.unsqueeze(0)

        B, N, _ = src.shape
        _, M, _ = dst.shape

        dist = - torch.matmul(src, dst.permute(0, 2, 1))

        if not normalized:
            dist *= 2
            dist += torch.sum(src ** 2, dim = -1)[:, :, None]
            dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
        else:
            dist += 1.0

        dist = torch.clamp(dist, min = 0.0, max=None)
        return dist
