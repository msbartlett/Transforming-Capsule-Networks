
import numpy as np
import torch
import torch.nn as nn

from pytorch_custom_utils import one_hot, Flatten


def squash(input_tensor, dim=-1):
    """
    Applies norm non-linearity (squash) to a capsule layer.

    Args:
      input_tensor: Input tensor. Shape is [batch, output_dim, output_atoms] for a
        fully connected capsule layer or
      dim: Dimensions to apply squashing function to.
    Returns:
      A tensor with same shape as input (rank 3) for output of this layer.
    """
    norm = torch.norm(input_tensor, dim=dim, keepdim=True)      # (B, Out_dim, 1)
    norm_squared = norm ** 2
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))


class Decoder(torch.nn.Module):
    """
     Given the last capsule output layer as input of shape [B, Num Classes, Output Atoms]
     add 3 fully connected layers on top of it.
     Feeds the masked output of the model to the reconstruction sub-network.
    """
    def __init__(self, in_channels, original_shape,
                 layer_sizes=(512, 1024)):

        super().__init__()

        self.original_shape = original_shape

        first_layer_size, second_layer_size = layer_sizes
        num_pixels = int(np.prod(original_shape))

        self.decoder = torch.nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, first_layer_size),
            # nn.BatchNorm1d(first_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(first_layer_size, second_layer_size),
            # nn.BatchNorm1d(second_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(second_layer_size, num_pixels),

            # nn.ReLU(inplace=True),
            # nn.Linear(second_layer_size, num_pixels),
        )

        # if the image has 1 channel then squash values to between 0 and 1
        if original_shape[0] == 1:
            self.decoder.add_module('6', nn.Sigmoid())

    def forward(self, capsule_embedding, labels, num_classes):
        """
        Given the last capsule output layer as input of shape [batch, 10, num_atoms]
        add 3 fully connected layers on top of it.
        Feeds the masked output of the model to the reconstruction sub-network.
        Args:
            capsule_embedding: tensor, output of the last capsule layer.
            labels: tensor, training labels
            num_classes: integer, number of classes being predicted
            # capsule_mask: tensor, for each data in the batch it has the one hot
            # encoding of the target id.

        Returns:
            The reconstruction images of shape [batch_size, num_pixels].
        """
        capsule_mask = one_hot(labels, num_classes).float()
        capsule_mask_3d = capsule_mask.unsqueeze(-1)
        atom_mask = capsule_mask_3d.expand_as(capsule_embedding)
        filtered_embedding = (capsule_embedding * atom_mask)
        reconstruction_2d = self.decoder(filtered_embedding)
        # reconstruction_3d = reconstruction_2d.view(-1, *self.original_shape)
        return reconstruction_2d
