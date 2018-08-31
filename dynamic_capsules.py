
"""
PyTorch implementation of Sabour et al. (2018)'s CapsNet model with dynamic routing-by-agreement.
Residual pose-to-vote transformations, sigmoid routing activation and euclidean and cosine agreement
measures have been added. The code is based off https://github.com/XifengGuo/CapsNet-Pytorch and Sara
Sabour's official release https://github.com/Sarasra/models/tree/master/research/capsules.

Author: Myles Bartlett, E-mail: `mb715@sussex.ac.uk`
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from loss import MarginLoss
from pytorch_custom_utils import truncated_normal_
from capsule_utils import squash
from transformations import ResidualBlock, InvertedResidualBlock


class Capsule(torch.nn.Module):
    """
    Base class for capsule layers.
    Implements dynamic routing-by-agreement.
    """
    def __init__(self, input_atoms, output_atoms,
                 input_dim, output_dim, num_routing=3,
                 resid_tform=False, bn_squash=False):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_atoms = input_atoms
        self.output_atoms = output_atoms
        self.num_routing = num_routing
        self.resid_tform = resid_tform
        self.bn_squash = bn_squash
        self.squash_func = self._bn_squash if bn_squash else squash

        # The paper itself does not mention the use of a bias term
        # however it is included in the official code released by Sara.
        self.biases = torch.nn.Parameter(
            torch.empty(output_dim, 1, output_atoms))
        # Bias initializer
        nn.init.constant_(self.biases, 0.1)
        # for normalizing activations before applying element-wise sigmoid
        self.bn_activation = nn.BatchNorm2d(self.output_dim)

    def _bn_squash(self, activations):
        """
        Squash the activations using batch-normalization
        followed by element-wise sigmoid.
        Args:
            activations:

        Returns:

        """
        return F.sigmoid(self.bn_activation(activations))

    def _update_routing(self, predictions):
        """
        Sums over scaled votes and applies squash to compute the activations.
        Iteratively updates routing logits (scales) based on the similarity between
        the activation of this layer and the votes of the layer below.
        Args:
            predictions: the prediction uj|i made by capsule i
        Returns:
            vector output of capsule j
        """
        detached = predictions.detach()
        logits = predictions.new_zeros(predictions.shape[:-1])

        for r in range(self.num_routing):
            coupling = F.softmax(logits, dim=1)
            if r == self.num_routing - 1:
                activations = (coupling.unsqueeze(-1) * predictions).sum(dim=-2, keepdim=True)
            else:
                activations = (coupling.unsqueeze(-1) * detached).sum(dim=-2, keepdim=True)

            votes = self.squash_func(activations + self.biases)
            if r < self.num_routing - 1:
                # Different agreement measures.
                # agreement = F.cosine_similarity(votes, detached, dim=-1)  # cosine similarity
                agreement = F.pairwise_distance(votes, detached, keepdim=True).sum(dim=-1)  # euclidean distance
                # agreement = (votes * detached).sum(dim=-1)    # scalar-product
                logits = logits + agreement

        return votes.squeeze(-2)

    def forward(self, input_tensor):
        raise NotImplementedError


class _PrimaryCaps(Capsule):
    """
    The primary capsules are the lowest level of multi-dimensional entities
    and, from an inverse graphics perspective, activating the primary capsules
    corresponds to inverting the rendering process.

    In total PrimaryCapsules has [32 × 6 × 6] capsule outputs (each output is an 8D vector)
    and each capsule in the [6 × 6] grid is sharing their weights with each other.

    We have routing only between two consecutive capsule layers (e.g. PrimaryCapsules
    and DigitCaps). Since Conv1 output is 1D, there is no orientation in its space to
    agree on. Therefore, no routing is used between Conv1 and PrimaryCapsules.
    """
    def __init__(self, input_atoms, output_atoms,
                 input_dim, output_dim, kernel_size,
                 stride, padding, num_routing=1,
                 resid_tform=True, bn_squash=True):

        super().__init__(input_atoms, output_atoms,
                         input_dim, output_dim, num_routing,
                         resid_tform, bn_squash)

        self.kernel_size = kernel_size
        self.stride = stride
        self.num_routing = num_routing
        self.resid_tform = resid_tform

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_atoms,
                      out_channels=output_dim * output_atoms,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
        )
        # we can just incorporate the BN/sigmoid activation into the sequential module
        # since they're are not dimension specific
        if self.bn_squash:
            self.conv.add_module('bn', nn.BatchNorm2d(output_dim * output_atoms))
            self.conv.add_module('logistic', nn.Sigmoid())

    def forward(self, input_tensor):
        predictions = self.conv(input_tensor)
        # split output channels
        if not self.resid_tform:
            predictions = predictions.view(-1, self.output_dim, self.output_atoms, *predictions.shape[-2:])
            predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()   # [B, Out dim, H, W, Out atoms]
            predictions = predictions.view(input_tensor.size(0), -1, self.output_atoms)    # [B, H x W x Out dim, Out atoms]
            if not self.bn_squash:
                # squashing non-linearity
                predictions = self.squash_func(predictions)

        return predictions


class _DigitCaps(Capsule):
    """
    A fully connected capsule layer.
    The length of the activity vector of each capsule in DigitCaps layer
    indicates presence of an instance of each class and is used to calculate the
    classification loss.
    """
    def __init__(self, input_atoms, output_atoms,
                 input_dim, output_dim, in_capsules,
                 num_routing=3, resid_tform=True,
                 bn_squash=True):

        super().__init__(input_atoms, output_atoms,
                         input_dim, output_dim, num_routing,
                         resid_tform, bn_squash)

        if resid_tform:
            self.resid_block = InvertedResidualBlock(in_channels=in_capsules * input_atoms,
                                                     out_channels=in_capsules * output_dim * output_atoms,
                                                     expansion_factor=6,
                                                     kernel_size=3,
                                                     stride=1)
            # self.resid_block = ResidualBlock(in_planes=in_capsules * input_atoms,
            #                                  planes=in_capsules * output_dim * output_atoms)
            # pose-to-vote transformation method
            self.vector_tform = self._resid_transform
        else:
            # Parameter initialization not specified in the paper
            self.weights = nn.Parameter(
                torch.empty(output_dim, input_dim, output_atoms, input_atoms))
            # Weights initializer
            truncated_normal_(self.weights.data)
            # pose-to-vote transformation method
            self.vector_tform = self._matrix_transform

    def _resid_transform(self, input_tensor):
        """
        Args:
            input_tensor: pose vector, shape [B, Input dim x Input atoms, H, W]

        Returns:
            predictions, uj, for each parent capsule (out_dim)
        """
        predictions = self.resid_block(input_tensor)     # [B, Input dim x Output dim x Output caps, H, W]
        predictions = predictions.view(predictions.size(0), self.output_dim, self.output_atoms, -1)
        predictions = predictions.permute(0, 1, 3, 2)
        return predictions

    def _matrix_transform(self, input_tensor):
        return (self.weights @ input_tensor[:, None, :, :, None]).squeeze(dim=-1)

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: tensor, input from primary capsule layer, (N, H, W, C)
        Returns:
            The length of the activity vector of each capsule in DigitCaps layer
             indicates presence of an instance of each class
        """
        predictions = self.vector_tform(input_tensor)
        votes = self._update_routing(predictions)
        return votes


class DynamicCapsules(nn.Module):
    """
    3-layer CapsNet model consisting of a feature extraction layer, followed
    by a primary capsule and digit capsule layer.
    """
    def __init__(self, input_shape, num_features=256, num_classes=10, num_routing=3,
                 resid_tform=False, bn_squash=True, res_features=False):
        super().__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_routing = num_routing
        self.resid_tform = resid_tform
        self.bn_squash = bn_squash

        # ReLU convolution layer
        # This layer converts pixel intensities to the activities of local feature
        # detectors that are then used as inputs to the primary capsule
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
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=input_shape[0], out_channels=num_features,
                          kernel_size=9, stride=1, padding=0),
                nn.BatchNorm2d(num_features=num_features, eps=0.001,
                               momentum=0.1, affine=True),
                nn.ReLU(inplace=True),
            )

        # Calculate input size after convolution
        output_size = input_shape[1]
        for layer in self.conv1.children():
            if isinstance(layer, nn.Conv2d):
                K = layer.kernel_size[0]
                s = layer.stride[0]
                output_size = int((output_size - K + 1) / s)

        self.primary_caps = _PrimaryCaps(input_atoms=num_features,
                                         output_atoms=16,    # 8
                                         input_dim=1,
                                         output_dim=8,     # 32
                                         kernel_size=9,
                                         stride=2,
                                         padding=0,
                                         resid_tform=resid_tform,
                                         bn_squash=bn_squash)

        # Recalculate input size after convolution
        output_size = int((output_size - self.primary_caps.kernel_size + 1)
                          / self.primary_caps.stride)
        input_dim = output_size * output_size * self.primary_caps.output_dim

        self.digit_caps = _DigitCaps(input_atoms=self.primary_caps.output_atoms,
                                     output_atoms=16,   # 16
                                     input_dim=input_dim,
                                     in_capsules=self.primary_caps.output_dim,
                                     output_dim=num_classes,
                                     num_routing=num_routing,
                                     resid_tform=resid_tform,
                                     bn_squash=bn_squash)

    def forward(self, x, labels=None):
        features = self.conv1(x)
        capsule_outputs = self.primary_caps(features)
        votes = self.digit_caps(capsule_outputs)

        # only makes sense to use the vector norm with the squashing
        # non-linearity
        if self.bn_squash:
            output = F.softmax(votes.sum(dim=-1), dim=-1)
        else:
            output = votes.norm(dim=-1)

        return output, votes


# For testing
if __name__ == '__main__':
    from capsule_utils import Decoder
    num_classes = 10

    batch_size = 9
    device = torch.device('cuda')

    test = torch.randn(batch_size, 1, 28, 28).to(device)
    input_shape = tuple(test.shape[1:])
    target = torch.LongTensor(batch_size).random_(0, 1).to(device)

    model = DynamicCapsules(input_shape=test.shape[1:]).to(device)
    decoder = Decoder(in_channels=num_classes * model.digit_caps.output_atoms,
                      original_shape=test.shape[1:]).to(device)

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    print("Total number of parameters:", sum(param.numel() for param in model.parameters()))

    print('Forward pass...')
    output, poses = model(test)
    remake = decoder(poses, target, num_classes)

    # criterion(output, pose_out, x=em-one, svhn, inv-resid, pc=8, no-remake, routing=3, y=target, r=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.01)
    # criterion = nn.CrossEntropyLoss()
    criterion = MarginLoss(num_classes=num_classes, reconstruction=False)

    optimizer.zero_grad()
    print("Calculating loss...")
    # loss = criterion(output, target)
    loss, _, _ = criterion(output, target)
    print('Computing gradients...')
    loss.backward()
    optimizer.step()
