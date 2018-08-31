
"""
Implementation of CapsNet with EM-routing from "Matrix Capsules with EM Routing".
Credit to Yang Lei (https://github.com/yl-1993/Matrix-Capsules-EM-PyTorch, 2018) for the
base code that we adapted for our purposes.

Original paper: https://openreview.net/pdf?id=HJWLfGWRb
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformations import ResidualBlock, InvertedResidualBlock


class _PrimaryCaps(nn.Module):
    """
    Each capsule contains a 4x4 pose matrix and an activation value.
    We use the regular convolution layer to implement the PrimaryCaps.
    We group 4×4+1 neurons to generate 1 capsule (pose matrix + activation).
    """
    def __init__(self, input_dim=32, out_capsules=32,
                 kernel_size=1, stride=1, padding=0, p_size=16):

        super().__init__()

        self.input_dim = input_dim
        self.out_capsules = out_capsules
        self.kernel_size = kernel_size
        self.stride = stride
        self.p_size = p_size
        # channel index at which to split the pose (4x4) and activation (1) vectors
        self._split_at = -out_capsules

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dim,
                      out_channels=out_capsules * (self.p_size + 1),
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
        )

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
    def __init__(self, in_capsules=32, out_capsules=32, kernel_size=3, stride=2,
                 num_routing=3, coordinate_addition=False, share_transform=False,
                 resid_tform=False, p_size=16, in_p_size=16):
        """
        Args:
            in_capsules: Dimensionality of input data
            out_capsules: Dimensionality of output. Equal to number of capsules
            kernel_size: Kernel size (symmetric)
            stride: Stride length
            num_routing: Number of EM-routing iterations to perform.
            coordinate_addition: Whether to include coordinate addition (used with class capsules)
            share_transform: Whether to share information the transformation matrices
            between different positions of the same capsule type (used with class capsules)
        """
        super().__init__()

        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_routing = num_routing
        self.coordinate_addition = coordinate_addition
        self.share_transform = share_transform
        self.resid_tform = resid_tform

        self.p_dim = p_size ** 0.5
        self.p_size = p_size

        if not self.resid_tform:
            if self.p_dim % 1 != 0:
                raise ValueError("Matrix size must be a square number.")
        # CONSTANTS
        self.eps = 1.e-08     # For stability when dividing/taking logs
        self._ln_2pi = torch.tensor(math.log(2 * math.pi)).cuda()
        self._lambda_0 = 1.e-03    # Initial inverse temperature value
        self._lambda_delta = 1.e-04  # lambda annealing step size

        # LEARNED PARAMETERS
        # If we do not activate a capsule, we must pay a fixed cost of −βu per data-point for
        #  describing the poses of all the lower-level capsules that are assigned to the higher-level
        # capsule.
        self.beta_u = nn.Parameter(torch.empty(out_capsules))
        nn.init.constant_(self.beta_u, 0.1)
        #  −βa is the cost of describing the mean and variance of capsule j.
        self.beta_a = nn.Parameter(torch.empty(out_capsules))
        nn.init.constant_(self.beta_a, 0.1)
        # 4x4 Trainable transformation matrix that is learned discriminatively.
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
            self.weights = nn.Parameter(torch.empty(1, kernel_size * kernel_size * in_capsules,
                                                    out_capsules, self.p_dim, self.p_dim))
            nn.init.normal_(self.weights.data)  # normal seems to work best out the schemes tested

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
        output = output[:, :, :, tile_filter, :]   # [B, K, H_out, K, W_out, L+1 x (16 + 1)]
        # Transpose 2nd and 3rd dimensions
        output = output.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, K, K, H_out, W_out, L+1 x (16 + 1)]

        return output, out_height, out_width

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
        batched_size = poses.size(0)    # B x H x W
        poses = poses.view(batched_size, -1, 1, self.p_dim, self.p_dim)

        weights = self.weights
        if self.share_transform:
            in_out_ratio = int(poses.size(1) / weights.size(1))
            # Tile the weights; can't use expand because the dimension is not a singleton
            weights = weights.repeat(1, in_out_ratio, 1, 1, 1)

        # Tile weights and poses
        weights = weights.expand(batched_size, -1, -1, -1, -1)      # one weight for each sample in the batch
        # tile for output capsules
        poses = poses.expand_as(weights)      # [B x H x W, K x K x L, L+1, 4, 4]
        # Vij = Mi.Wij
        votes = poses @ weights
        votes = votes.view(batched_size, -1, self.out_capsules, self.p_size)    # [B x H x W, K x K x L, L+1, 16]
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

    def _em_routing(self, votes, activation_in):
        """
        Iterative adjusts the means, variances, and activation
        probabilities of the capsules in layer L + 1 and the assignment
        probabilities between all i ∈ ΩL, j ∈ ΩL+1.

        Args:
            votes: Votes for the parent capsules, [B x H_out x W_out, K x K x L, L+1, 16]
            activation_in: activations of the child capsules, [B x H_out x W_out, K x K x L, L+1, 1]

        Return:
            Activations and poses of the capsules in layer L + 1 given the activations
            and votes of capsules in layer L.
            poses: [B x H_out x W_out, 1, L+1, 1]
            activation_out: [B x H_out x W_out, L+1]
        """
        # Compute the assignment probability rij to quantify the connection between the
        # children capsules and the parent capsules.
        r = votes.new_full(size=votes.shape[:-1], fill_value=(1 / self.out_capsules))
        # inverse temperature
        lambda_ = self._lambda_0
        for t in range(self.num_routing):
            # TODO: Find better inverse temperature annealing schedule
            lambda_ += self._lambda_delta
            activation_out, mean, std_dev = self._m_step(activation_in, r, votes, lambda_)
            # In the last iteration, the M-step has already completed its final calculation
            if t < self.num_routing - 1:
                r = self._e_step(mean, std_dev, activation_out, votes)

        return mean, activation_out

    def _m_step(self, activation_in, r, votes, lambda_):
        """
        The M-step for each Gaussian consists of finding the mean
        of the weighted (by r_ij) datapoints and the variance about
        that mean. It holds the assignment probabilities
        constant and adjusts each Gaussian to maximize the sum of
        the weighted log probabilities that the Gaussian would generate
        the datapoints assigned to it.
        """
        r = r * activation_in
        # Normalize Rij by the dividing by its sum over all L+1 capsules
        r = r / (r.sum(dim=2, keepdim=True) + self.eps)
        # Sum over all child capsules and patches
        r_sum = r.sum(dim=1, keepdim=True)
        # Both mean and variance calculations require the normalized coefficient
        r_prob = (r / (r_sum + self.eps)).unsqueeze(-1)
        # Compute mean and standard deviation, summing over all capsules in L+1
        mean = (r_prob * votes).sum(dim=1, keepdim=True)
        std_dev = ((r_prob * (votes - mean) ** 2).sum(dim=1, keepdim=True)
                   + self.eps).sqrt()

        # Reshape for broadcasting
        r_sum = r_sum.view(-1, self.out_capsules, 1)
        std_dev = std_dev.view(-1, self.out_capsules, self.p_size)
        # cost of describing a datapoint, i, in layer L, for capsule j, in layer L+1
        cost = (self.beta_u.view(self.out_capsules, 1) + torch.log(std_dev)) * r_sum
        # Higher-level capsule’s activation probability
        activation_out = F.sigmoid(lambda_ * (self.beta_a - cost.sum(dim=-1)))
        std_dev = std_dev.unsqueeze(1)

        return activation_out, mean, std_dev

    def _e_step(self, mean, std_dev, activation_out, votes):
        """
        The E-step is used to determine, for each datapoint, the probability
        with which it is assigned to each of the Gaussians. These assignment
        probabilities, r_ij, act as weights.
        """
        # P_ijh is the probability density of the hth component of the vectorized vote
        # V_ij under j's Gaussian model for dimension h
        ln_p = - (votes - mean) ** 2 / (2 * std_dev ** 2) - torch.log(std_dev) - self._ln_2pi / 2
        # Since P is in log form, we also take the log of the activations before adding the two
        ln_ap = ln_p.sum(dim=3) + torch.log(activation_out.view(-1, 1, self.out_capsules))
        # Update routing assignment probability
        r = F.softmax(ln_ap, dim=-1)

        return r

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
        votes_shape = votes.shape
        votes = votes.view(-1, out_height, out_width, self.in_capsules,
                           self.out_capsules, self.p_size)
        # Input coordinates scaled between 0 and (1 - in_height)
        patch_h = torch.arange(out_height) / out_height
        patch_w = torch.arange(out_width) / out_width

        row_coords = votes.new_zeros(1, out_height, 1, 1, 1, self.p_size)
        col_coords = votes.new_zeros(1, 1, out_width, 1, 1, self.p_size)

        # add the scaled coordinate (row, column) of the center of the receptive field of each
        # capsule to the first two elements of the right-hand column of its vote matrix
        row_coords[0, :, 0, 0, 0, 0] = patch_h      # first element (0)
        col_coords[0, 0, :, 0, 0, 1] = patch_w      # second element (1)

        votes = votes + row_coords + col_coords
        votes = votes.view(votes_shape)     # [B, H_in x W_in x D_in, D_out, 16]
        return votes

    def forward(self, input_tensor):
        batch_size, in_height, in_width = input_tensor.shape[:3]
        # Convolutional capsule
        if not self.share_transform and not self.resid_tform:
            # Add patches
            output, out_height, out_width = self._kernel_tile(input_tensor)
            # Extract the poses
            poses = output[..., :-self.in_capsules].contiguous()
            poses = poses.view(-1, self.kernel_size * self.kernel_size * self.in_capsules, self.p_size)
            # Extract the activations
            activations = output[..., -self.in_capsules:].contiguous()        # [B x H_out x W_out, K x K x L, 1]
            activations = activations.view(-1, self.kernel_size * self.kernel_size * self.in_capsules, 1)
            # Generate votes for each parent capsule
            votes = self.pose_tform(poses)    # [B x H_out x W_out, K x K x L, L+1, 16]
            # EM-routing procedure
            poses, activations = self._em_routing(votes, activations)
            poses = poses.view(batch_size, out_height, out_width, self.out_capsules * self.p_size)
            activations = activations.view(batch_size, out_height, out_width, self.out_capsules)
            output = torch.cat([poses, activations], dim=-1)        # [B, W_out, H_out, 16 * L+1]

        # Class capsule
        else:
            # Extract the poses
            poses = input_tensor[..., :-self.in_capsules].contiguous()        # [B, H_out, W_out, 16 * L+1]
            # Extract the activations
            activations = input_tensor[..., -self.in_capsules:].contiguous()  # [B, H_out x W_out, 16 * 1]
            activations = activations.view(batch_size, -1, 1)
            # Generate the votes for each of the parent capsules.
            votes = self.pose_tform(poses)
            # Coordinate addition
            if self.coordinate_addition:
                votes = self._coord_addition(votes, in_height, in_width)
            # EM-routing procedure
            poses, output = self._em_routing(votes, activations)
            poses = poses.squeeze(1)

        return output, poses


class EMCapsules(nn.Module):
    """
      EM-Capsule Model.
      """
    def __init__(self, input_shape, num_features=256, num_prime_caps=8,
                 conv_caps_layers=None, num_classes=10, num_routing=2,
                 res_features=False, resid_tform=True, skip_connection=False):
        """
        Args:
            input_shape: tuple or list, shape of the input image [C, H, W]
            num_features: integer, Output channels of the ReLU convolutional layer
            num_prime_caps: integer, number of capsule units in the Primary Capsule layer
            C: integer, number of capsule units in the first convolutional capsule layer
            D: integer, number of capsule units in the second convolutional capsule layer
            num_classes: integer, number of classes to predict.
            num_routing: integer, number of routing iterations to perform
            remake: bool, whether to add reconstruction
        """
        super().__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

        if res_features:
            first_layer, second_layer, third_layer = 64, 256, 512
            self.conv1 = nn.Sequential(
                # nn.Conv2d(in_channels=input_shape[0], out_channels=first_layer,
                #           kernel_size=9, stride=1, padding=0),
                # nn.BatchNorm2d(num_features=first_layer, eps=0.001,
                #                momentum=0.1, affine=True),
                # nn.ReLU(inplace=True),

                ResidualBlock(in_planes=input_shape[0], planes=first_layer, stride=1),
                ResidualBlock(in_planes=first_layer, planes=second_layer, stride=2),
                ResidualBlock(in_planes=second_layer, planes=third_layer, stride=1),
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
                                         stride=1, padding=0, p_size=16)
        in_capsules = num_prime_caps
        self.conv_caps = nn.ModuleList()
        if conv_caps_layers:
            for layer in conv_caps_layers:
                # layer is a tuple of out_capsules, kernel_size, stride
                self.conv_caps.append(_ConvCaps(in_capsules, out_capsules=layer[0],
                                                kernel_size=layer[1], stride=layer[2],
                                                num_routing=num_routing, share_transform=False,
                                                coordinate_addition=False, resid_tform=False,
                                                p_size=16))
                in_capsules = layer[0]

        in_p_size = self.primary_caps.p_size
        self.class_caps = _ConvCaps(in_capsules, num_classes, kernel_size=1, stride=1,
                                    num_routing=num_routing, coordinate_addition=True,
                                    share_transform=True, resid_tform=resid_tform,
                                    p_size=16, in_p_size=in_p_size)

        self.skip_connection = skip_connection
        if skip_connection:
            self.shortcut = nn.Sequential()
            out_capsules = self.conv_caps[-1].out_capsules if self.conv_caps \
                else self.primary_caps.out_capsules
            if num_features != 16 or self.primary_caps.stride > 1:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(num_features, out_capsules, kernel_size=self.primary_caps.kernel_size,
                              stride=self.primary_caps.stride, bias=False),
                    nn.BatchNorm2d(out_capsules)
                )

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

        activations, poses = self.class_caps(output)

        return activations, poses


def em_one(input_shape, num_classes, num_routing=2,
           res_features=True, resid_tform=True, skip=False):
    conv_caps = []
    return EMCapsules(input_shape=input_shape,
                      num_features=512,
                      num_prime_caps=8,
                      conv_caps_layers=conv_caps,
                      num_classes=num_classes,
                      num_routing=num_routing,
                      resid_tform=resid_tform,
                      res_features=res_features,
                      skip_connection=skip)


def em_standard(input_shape, num_classes, num_routing=2,
                res_features=False, resid_tform=True, skip=False):
    conv_caps = [(32, 3, 2), (32, 3, 1)]
    return EMCapsules(input_shape=input_shape, num_features=32, num_prime_caps=32,
                      conv_caps_layers=conv_caps, num_classes=num_classes,
                      num_routing=num_routing, resid_tform=resid_tform,
                      res_features=res_features, skip_connection=skip)


def em_small(input_shape, num_classes, num_routing=2,
             res_features=False, resid_tform=False, skip=False):
    conv_caps = [(16, 3, 2), (16, 3, 1)]
    return EMCapsules(input_shape=input_shape, num_features=64, num_prime_caps=8,
                      conv_caps_layers=conv_caps, num_classes=num_classes,
                      num_routing=num_routing, resid_tform=resid_tform,
                      res_features=res_features, skip_connection=skip)


if __name__ == '__main__':
    A, B, C, D = 64, 8, 16, 16
    num_classes = 10

    batch_size = 9
    device = torch.device('cuda')

    test = torch.randn(batch_size, 1, 28, 28).to(device)
    input_shape = tuple(test.shape[1:])
    target = torch.LongTensor(batch_size).random_(0, 1).to(device)

    model = em_small(input_shape, num_classes=num_classes).to(device)
    # model = em_one(input_shape, num_classes=num_classes,
    #                  resid_tform=True, skip=True).to(device)

    print("=> Total number of parameters:", sum(param.numel() for param in model.parameters()))

    print('=> Forward pass...')
    output, poses = model(test)

