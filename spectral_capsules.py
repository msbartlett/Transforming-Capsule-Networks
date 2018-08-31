
"""
PyTorch implementation of S-Capsules from "Spectral Capsule Networks".
The paper leaves some aspects open to interpretation, and the network was
applied to time-series analysis rather than image classification, there are
some discrepancies. The code is based on that used for E-Capsules, with the
routing algorithm swapped out and a shortcut connection introduced.

S-Capsules paper: https://openreview.net/forum?id=HJuMvYPaM

Author: Myles Bartlett, E-mail: `mb715@sussex.ac.uk`
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from loss import SpreadLoss
from transformations import ResidualBlock, InvertedResidualBlock
from pytorch_custom_utils import truncated_normal_


# Computes the dominant right singular value, Vj
def top1_svd(m):
    """
    This function only computes the dominant right singular
    value and does not waste time with U, S as full SVD would.
    """
    m_gram = m.t() @ m
    e, v = torch.symeig(m_gram, eigenvectors=True)
    _, top1 = torch.max(e, 0)   # axis preserving the most variance
    return v[:, top1]


class _PrimaryCaps(nn.Module):
    """
    Each capsule contains a 4x4 pose matrix and an activation value.
    We use the regular convolution layer to implement the PrimaryCaps.
    We group 4×4+1 neurons to generate 1 capsule (pose matrix + activation).
    """
    def __init__(self, input_dim=256, out_capsules=8, kernel_size=1, stride=1,
                 p_size=16, padding=0):

        super().__init__()

        self.input_dim = input_dim
        self.out_capsules = out_capsules
        self.kernel_size = kernel_size
        self.stride = stride
        self.p_size = p_size
        # channel index at which to split the pose (4x4) and activation (1) vectors
        self._split_at = -out_capsules

        # self.conv = ResidualBlock(in_planes=input_dim,
        #                           planes=out_capsules * (p_dim * p_dim + 1),
        #                           stride=stride)
        self.conv = nn.Conv2d(in_channels=input_dim,
                              out_channels=out_capsules * (self.p_size + 1),
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

    def forward(self, x):
        """
        Args:
            x: input of shape (batch size, in channels,
            input height, input width)
        Returns:
            L+1 output capsules, [B, H_out, W_out, L+1 * (4 * 4 + 1)]
        """
        output = self.conv(x)
        # apply sigmoid non-linearity to the activation logits
        output[:, self._split_at:] = F.sigmoid(output[:, self._split_at:])
        output = output.permute(0, 2, 3, 1).contiguous()   # [B, H_out, W_out, L+1 * (4 * 4 + 1)]
        return output


class _ConvCaps(nn.Module):
    """
    Create a convolutional capsule layer
    """
    def __init__(self, in_capsules=8, out_capsules=16, p_size=16, in_p_size=16, kernel_size=1, stride=1,
                 eta=1, share_transform=True, coordinate_addition=False, resid_tform=True):
        """
        Args:
            in_capsules: Dimensionality of input data
            out_capsules: Dimensionality of output. Equal to number of capsules
            eta: learning rate used during the spectral routing procedure
            coordinate_addition: Whether to include coordinate addition (used with class capsules)
        """
        super().__init__()

        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.coordinate_addition = coordinate_addition
        self.share_transform = share_transform
        self.kernel_size = kernel_size
        self.stride = stride

        # CONSTANTS
        self.p_dim = p_size ** 0.5
        self.p_size = p_size

        if not resid_tform:
            if self.p_dim % 1 != 0:
                raise ValueError("Matrix size must be a square number.")

        self.eps = 1.e-07   # For stability when dividing/taking logs; same as numpy default
        self.eta = eta      # eta is annealed during training

        # LEARNED PARAMETERS
        # activation bias (referred to as b in the paper) is discriminatively trained
        # and η is linearly annealed during training.
        self.activation_bias = nn.Parameter(torch.empty(out_capsules))
        nn.init.constant_(self.activation_bias, 0.1)

        if resid_tform:
            # Use a residual block to generate the votes
            self.pose_tform = self._resid_transform
            # self.resid_block = InvertedResidualBlock(in_channels=in_capsules * in_p_size,
            #                                          out_channels=out_capsules * in_capsules * self.p_size,
            #                                          stride=1,
            #                                          expansion_factor=6)
            self.resid_block = ResidualBlock(in_planes=in_capsules * in_p_size,
                                             planes=out_capsules * in_capsules * self.p_size)
        else:
            # Use a 4x4 weight matrix to generate the votes
            self.pose_tform = self._matrix_transform
            self.weights = nn.Parameter(torch.empty(kernel_size * kernel_size * in_capsules,
                                                    out_capsules, self.p_dim, self.p_dim))
            # truncated_normal_(self.weights.data)
            nn.init.orthogonal_(self.weights.data)
            self.weights.data = self.weights.data.unsqueeze(dim=0)  # some initializers do not accept the batch dim

    def _matrix_transform(self, poses):
        """
        Compute votes for the parent capsules.
        For ConvCaps:
            Input:     [B x H x W, K x K x L, 16]
            Output:    [B x H x W, K x K x L, L+1, 16]
        For ClassCaps:
            Input:     [B, H, W, L x 16]
            Output:    [B, H x W x L, L+1, 16]
        """
        batched_size = poses.size(0)
        poses = poses.view(batched_size, -1, 1, self.p_dim, self.p_dim)

        weights = self.weights
        if self.share_transform:
            in_out_ratio = int(poses.size(1) / weights.size(1))
            # Tile the weights; can't use expand because the dimension is not a singleton
            weights = weights.repeat(1, in_out_ratio, 1, 1, 1)

        # Tile weights and poses
        weights = weights.expand(batched_size, -1, -1, -1, -1)      # one weight for each sample in the batch
        poses = poses.expand_as(weights)      # [B x H x W, K x K x L, L+1, 4, 4]
        # Vij = Mi.Wij
        votes = poses @ weights
        # [B x H x W, K x K x L, L+1, 16]
        votes = votes.view(batched_size, poses.size(1), self.out_capsules, self.p_size)
        return votes

    def _resid_transform(self, poses):
        """
        Input:     [B, H, W, L x 16]

        Output:    [B, H x W x L, L+1, 16]
        """
        poses = poses.permute(0, 3, 1, 2)
        votes = self.resid_block(poses)
        # [B, L1 * L * 16, H, W]
        votes = votes.view(poses.size(0), self.out_capsules, self.p_size, -1)
        votes = votes.permute(0, 3, 1, 2)
        return votes

    # Deprecated: this function is superseded by 'top1_svd'
    @staticmethod
    def _svd_to_pose(vote):
        """
        The pose vector for capsule j is simply the first (dominant)
        right singular vector uj = V[0, :], which is the normal vector
        of a linear subspace preserving most of the variance in the vote
         vectors of the capsules in layer L.
        Args:
            vote: tensor, votes from layer L for capsule j in layer L+1.
            [K x K x L, 16]
        Returns:
            pose: tensor, pose vector for capsule j in layer L+1.
            [16]
        """
        # svd with cuda is VERY slow
        _, _, v = torch.svd(vote)
        pose = v[:, 0]  # the poses are the values from the principal axis of v
        return pose

    def _spectral_routing(self, votes, activations):
        """
        Args:
            votes: tensor, votes for the pose of each higher-level capsule
            [B x H_in x W_in, K x K x L, L+1, 16]
            activations: tensor, activations of the capsules in the layer below
            [B x H_in x W_in, K x K x L, L+1, 1

        Returns:
            poses: poses of the higher level capsules (L+1)
            [B x H_out x W_out, L+1, 16]
            activations: activations for L+1
            [B x H_out x W_out, L+1]
        """
        # Weight the votes by their corresponding activations
        votes = activations.unsqueeze(-1) * votes
        # permute the L and L+1 dimensions so we can easily iterate
        # through the votes matrices with svd and to prepare to matmul
        votes = votes.permute(0, 2, 1, 3).contiguous()   # [B x H_in x W_in, L+1, K x K x L, 16]
        # For each capsule j in layer L+1, iterate over its weighted pose matrix Yj
        #  and use spectral decomposition to obtain the pose matrix for j.
        poses = [top1_svd(Yj.detach())
                 for Yj in torch.unbind(votes.view(-1, *votes.shape[2:]).cpu(), dim=0)]

        # since svd is computed on the cpu, poses need to be converted back to cuda
        poses = torch.stack(poses, dim=0).to(votes.device)  # [B x H_out x  W_out x L+1, 16]
        poses = poses.view(-1, votes.size(1), self.p_size)  # [B x H_out x W_out, L+1, 16]
        yu = votes @ poses.unsqueeze(-1)  # [B x H_out x W_out, L+1, L, 16] . [B x H_out x W_out, L+1, 16, 1]
        yu = yu.squeeze(-1)  # [B x H_out x W_out, L+1, L]

        yu_norm_sq = (yu ** 2).sum(-1)  # L2 norm squared, [B x H_out x W_out, L+1]
        y_fro_sq = (votes ** 2).sum(-1).sum(-1)  # Frobenius norm squared, [B x H_out x W_out, L+1]
        # sigmoid(eta [||Yj uj||^2 / ||Yj|^2 - b])
        activations = self.eta * (yu_norm_sq / y_fro_sq - self.activation_bias)  # [B x H_out x W_out, L+1]
        activations = F.sigmoid(activations)
        return poses, activations

    def _kernel_tile(self, input_tensor):
        """
        The routing procedure is used between each adjacent pair of capsule layers.
        For convolutional capsules, each capsule in layer L+1 sends feedback only
        to capsules within its receptive field in layer L. Therefore each convolutional
        instance of a capsule in layer L receives at most kernel size X kernel size feedback
        from each capsule type in layer L+1.

        Args:
            Output of previous capsule layer, [B, H_in, W_in, L+1 x (16 + 1)]
        Returns:
            Tiled input tensor, [B, K, K, H_out, W_out, L+1 x (16 + 1)]
        """
        in_height, in_width = input_tensor.shape[1:3]
        # Calculate the output width/height (symmetric kernel) of the convolution
        out_height = int((in_height - self.kernel_size + 1) / self.stride)
        out_width = int((in_width - self.kernel_size + 1) / self.stride)
        # Image boundary relative to kernel size
        end = in_width - self.kernel_size + 1
        # For each kernel get the patch indexes
        # Since the indexes are 2-d, the tensor will be tiled
        tile_filter = [[(step_idx + k_idx) for step_idx in range(0, end, self.stride)]
                       for k_idx in range(0, self.kernel_size)]
        # Rows
        output = input_tensor[:, tile_filter, :, :]
        # Cols
        output = output[:, :, :, tile_filter, :]  # [B, K, H_out, K, W_out, L+1 x (16 + 1)]
        # Transpose 2nd and 3rd dimensions
        output = output.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, K, K, H_out, W_out, L+1 x (16 + 1)]

        return output, out_height, out_width

    def _coord_addition(self, votes, out_height, out_width):
        """
        Coordinate addition. Adds the scaled coordinate (row, column) of the center of the
        receptive field of each capsule to the first two elements of the right-hand column
        of its vote matrix. This should encourage the shared final transformations to produce
        values for those two elements that represent the fine position of the entity relative
        to the center of the capsule’s receptive field.

        Args:
            votes: tensor, votes for the pose matrix of capsules in L+1
            out_height: integer, height of the output
            out_width, integer, width of the output

        Returns:
            votes: tensor, vote matrix with scaled coordinates of the center of the
            receptive field of each capsule added to the first two elements of the
            right-hand column.
            Same shape as input tensor, [B, H_in x W_in x L, L+1, 16]
        """
        input_shape = votes.shape
        votes = votes.view(-1, out_height, out_width, self.in_capsules,
                           self.out_capsules, self.p_size)
        # Input coordinates scaled between 0 and (1 - in_height)
        patch_h = torch.arange(out_height) / out_height
        patch_w = torch.arange(out_width) / out_width

        row_coords = votes.new_zeros(1, out_height, 1, 1, 1, self.p_size)
        col_coords = votes.new_zeros(1, 1, out_width, 1, 1, self.p_size)

        # add the scaled coordinate (row, column) of the center of the receptive field of each
        # capsule to the first two elements of the right-hand column of its vote matrix
        row_coords[0, :, 0, 0, 0, 0] = patch_h  # first element (0)
        col_coords[0, 0, :, 0, 0, 1] = patch_w  # second element (1)

        votes = votes + row_coords + col_coords
        votes = votes.view(input_shape)  # [B, H_in x W_in x D_in, D_out, 16]
        return votes

    def forward(self, input_tensor):

        batch_size, in_height, in_width = input_tensor.shape[:3]

        if not self.share_transform:
            output, out_height, out_width = self._kernel_tile(input_tensor)
            poses = output[..., :-self.in_capsules].contiguous()
            poses = poses.view(-1, self.kernel_size * self.kernel_size * self.in_capsules, 16)
            # Extract the activations
            activations = output[..., -self.in_capsules:].contiguous()  # [B x H_out x W_out, K x K x L, 1]
            activations = activations.view(-1, self.kernel_size * self.kernel_size * self.in_capsules, 1)
            # Generate votes for each parent capsule
            votes = self.pose_tform(poses)  # [B x H_out x W_out, K x K x L, L+1, 16]
            poses, activations = self._spectral_routing(votes, activations)
            poses = poses.view(batch_size, out_height, out_width, self.out_capsules * 16)
            activations = activations.view(batch_size, out_height, out_width, self.out_capsules)
            output = torch.cat([poses, activations], dim=-1)  # [B, W_out, H_out, 16 * L+1]

        else:
            poses = input_tensor[..., :-self.in_capsules].contiguous()        # [B, H_out x W_out, 16 * L+1]
            activations = input_tensor[..., -self.in_capsules:].contiguous()  # [B, H_out x W_out, 16 * 1]
            activations = activations.view(batch_size, -1, 1)
            # Generate the votes for each of the parent capsules.
            votes = self.pose_tform(poses)
            # votes = self._compute_votes(poses, self.weights)
            # Coordinate addition
            if self.coordinate_addition:
                votes = self._coord_addition(votes, in_height, in_width)
            # EM-routing procedure
            poses, activations = self._spectral_routing(votes, activations)
            output = activations
            poses = poses.view(batch_size, self.out_capsules, self.p_size)

        return output, poses

# class ConvBlock(nn.Module):
#
#     def __init__(self, inplanes, planes, kernel_size, stride):


class SpectralCapsules(nn.Module):
    """
    Spectral capsule network.
    """
    def __init__(self, input_shape, num_features=256, num_prime_caps=8,
                 conv_caps_layers=None, num_classes=10, eta_0=1,
                 p_size=16, res_features=False, resid_tform=True,
                 skip_connection=True):
        """
        Args:
            input_shape: tuple or list, shape of the input image [C, H, W]
            num_features: integer, Output channels of the ReLU convolutional layer
            num_prime_caps: integer, number of capsule units in the Primary Capsule layer
            num_classes: integer, number of classes to predict.
        """
        super().__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.p_dim = p_size ** 0.5
        self.p_size = p_size
        self.eta_0 = eta_0

        if res_features:
            if res_features:

                first_layer, second_layer, third_layer = 64, 256, 512
                self.conv1 = nn.Sequential(
                    ResidualBlock(in_planes=input_shape[0], planes=first_layer, stride=1),
                    ResidualBlock(in_planes=first_layer, planes=second_layer, stride=2),
                    ResidualBlock(in_planes=second_layer, planes=num_features, stride=1),
                )
        else:
            # ReLU Convolution
            self.conv1 = nn.Sequential(

                nn.Conv2d(in_channels=input_shape[0], out_channels=num_features,
                          kernel_size=9, stride=1, padding=0),
                nn.BatchNorm2d(num_features=num_features, eps=0.001,
                               momentum=0.1, affine=True),
                nn.ReLU(inplace=False),
            )

        # Capsule layers
        self.primary_caps = _PrimaryCaps(num_features, num_prime_caps, kernel_size=9,
                                         stride=2, p_size=6, padding=0)

        in_capsules = num_prime_caps

        self.conv_caps = nn.ModuleList()
        if conv_caps_layers is not None:
            for layer in conv_caps_layers:
                # layer is a tuple of out_capsules, kernel_size, stride
                self.conv_caps.append(_ConvCaps(in_capsules, out_capsules=layer[0],
                                                kernel_size=layer[1], stride=layer[2],
                                                eta=eta_0, p_size=16, share_transform=False,
                                                coordinate_addition=False, resid_tform=False))
                in_capsules = layer[0]

        in_p_size = self.primary_caps.p_size
        self.output_caps = _ConvCaps(in_capsules, num_classes, eta=eta_0, p_size=8,
                                     in_p_size=in_p_size, share_transform=True,
                                     coordinate_addition=True, resid_tform=resid_tform)

        self.skip_connection = skip_connection
        if skip_connection:
            self.shortcut = nn.Sequential()
            out_capsules = self.conv_caps[-1].out_capsules if self. conv_caps\
                else self.primary_caps.out_capsules
            if num_features != out_capsules or self.primary_caps.stride > 1:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(num_features, out_capsules, kernel_size=self.primary_caps.kernel_size,
                              stride=self.primary_caps.stride, bias=False),
                    nn.BatchNorm2d(out_capsules)
                )

    def anneal_eta(self, delta):
        # return
        new_eta = self.output_caps.eta - delta
        new_eta = max(0.1 * self.eta_0, new_eta)
        for layer in self.conv_caps:
            layer.eta = new_eta
        self.output_caps.eta = new_eta

    def forward(self, x):
        features = self.conv1(x)
        output = self.primary_caps(features)

        for layer in self.conv_caps:
            output, _ = layer(output)

        if self.skip_connection:
            split_at = self.conv_caps[-1].out_capsules if self.conv_caps\
                else self.primary_caps.out_capsules
            residual = self.shortcut(features).permute(0, 2, 3, 1)
            output[..., -split_at:] += residual          # skip connection

        activations, poses = self.output_caps(output)

        return activations, poses


conv_layers = []


def spectral_capsules(input_shape, num_features=64, num_prime_caps=16,
                      conv_caps_layers=conv_layers, num_classes=10,
                      eta=1., p_size=16, resid_tform=True, skip=False,
                      res_features=False, num_routing=None):

    return SpectralCapsules(input_shape=input_shape, num_features=num_features,
                            num_prime_caps=num_prime_caps, conv_caps_layers=conv_caps_layers,
                            num_classes=num_classes, eta_0=eta, p_size=p_size,
                            resid_tform=resid_tform, res_features=res_features,
                            skip_connection=skip)


if __name__ == '__main__':
    import timeit
    num_classes = 10

    batch_size = 2
    device = torch.device('cuda')

    test = torch.randn(batch_size, 1, 28, 28).to(device)
    input_shape = tuple(test.shape[1:])
    target = torch.LongTensor(batch_size).random_(0, 1).to(device)

    model = spectral_capsules(input_shape).to(device)

    print("Total number of parameters:", sum(param.numel() for param in model.parameters()))

    print('Forward pass...')
    start = timeit.default_timer()
    output, poses = model(test)
    print(timeit.default_timer() - start)

    # criterion(output, pose_out, x=em-one, svhn, inv-resid, pc=8, no-remake, routing=3, y=target, r=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.01)
    # criterion = nn.CrossEntropyLoss()
    criterion = SpreadLoss(num_classes=num_classes, reconstruction=False)

    optimizer.zero_grad()
    print("Calculating loss...")
    # loss = criterion(output, target)
    loss, _, _ = criterion(output, target)
    print('Computing gradients...')
    loss.backward()
    optimizer.step()
    criterion.increment_margin()

