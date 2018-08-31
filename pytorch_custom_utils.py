
"""
Some useful functions for image processing with PyTorch.
"""

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.autograd import Variable


def img_parse(input_file, output_file, output_size):
    img = Image.open(input_file)
    img = img.resize((output_size, output_size))
    img.save(output_file)


def tensor_to_jpg(tensor, filename):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
    pil = tensor_to_pil(tensor)
    pil.save(filename)


def one_hot(y, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims and convert it
    to a 1-hot representation with n+1 dims.
    """

    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.long().view(-1, 1)

    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1

    y_one_hot = y.new_zeros(y_tensor.size(0), n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)

    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def truncated_normal_(tensor, std_dev=0.01):
    """
    Initialize tensor data from a truncated normal
    distribution.

    Args:
        tensor: Tensor to be initialized
        std_dev: Standard deviation to truncate at

    Returns:
        None: inplace operation, initializes the tensor
        with data drawn from a truncated normal distribution
    """
    tensor.zero_()
    tensor.normal_(std=std_dev)
    # Resample until all data is within 2 standard deviations
    while torch.sum(torch.abs(tensor) > 2 * std_dev) > 0:
        t = tensor[torch.abs(tensor) > 2 * std_dev]
        t.zero_()
        tensor[torch.abs(tensor) > 2 * std_dev] = torch.normal(t, std=std_dev)


class Flatten(nn.Module):
    """
    Flattens the input batch-wise, outputting
    a tensor of shape [B, -1]
    """
    def forward(self, x):
        return x.view(x.size(0), -1)
