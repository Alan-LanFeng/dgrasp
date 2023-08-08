import torch
import torch.nn as nn

def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()])
            else:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=False), nn.BatchNorm1d(mlp_channels[k]), nn.ReLU()])
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)


class PointNetEncoder_mtr(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None):
        super().__init__()
        self.pre_mlps = build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False
        )
        self.mlps = build_mlps(
            c_in=hidden_dim* 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False
        )

        if out_channels is not None:
            self.out_mlps = build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels],
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None

    def forward(self, points):
        """
        Args:
            points (batch_size, num_points, C):

        Returns:
        """
        batch_size, num_points, C = points.shape

        # Reshape for batch normalization
        points = points.reshape(-1, C)

        # pre-mlp
        points_feature = self.pre_mlps(points)  # (batch_size * num_points, C)

        # Reshape back to original shape
        points_feature = points_feature.view(batch_size, num_points, -1)

        # get global feature
        pooled_feature = points_feature.max(dim=1)[0]
        points_feature = torch.cat(
            (points_feature, pooled_feature[:, None, :].repeat(1, num_points, 1)), dim=-1)

        # Reshape for batch normalization
        points_feature = points_feature.view(-1, points_feature.shape[-1])

        # mlp
        feature_buffers = self.mlps(points_feature)

        # Reshape back to original shape
        feature_buffers = feature_buffers.view(batch_size, num_points, -1)

        # max-pooling
        feature_buffers = feature_buffers.max(dim=1)[0]  # (batch_size, C)

        # out-mlp
        if self.out_mlps is not None:
            feature_buffers = self.out_mlps(feature_buffers)
        return feature_buffers

