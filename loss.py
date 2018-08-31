
"""
Loss functions written to be compatible with CapsNet models.
Includes classes for Cross-Entropy, Spread (Hinton et al., 2018),
and Margin (Sabour et al., 2017) losses.

Author: Myles Bartlett, E-mail: `mb715@sussex.ac.uk`
"""

from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from pytorch_custom_utils import one_hot


class CrossEntropyLoss(_Loss):
    """
    Standard cross-entropy loss with optional reconstruction loss included
    """
    def __init__(self, num_classes, reconstruction):

        super().__init__()

        self.num_classes = num_classes
        self.reconstruction = reconstruction
        if reconstruction:
            self.reconstruction_loss = ReconstructionLoss()

    def forward(self, logits, labels, reconstruction=None, image=None):

        classification_loss = F.cross_entropy(logits, labels)
        all_loss = classification_loss
        reconstruction_loss = None
        if self.reconstruction:
            reconstruction_loss = self.reconstruction_loss(image, reconstruction)
            all_loss += reconstruction_loss

        return all_loss, classification_loss, reconstruction_loss


class SpreadLoss(_Loss):
    """
    In order to make the training less sensitive to the initialization and hyper-parameters
    of the model, we use “spread loss” to directly maximize the gap between the activation of
    the target class (at) and the activation of the other classes. If the activation of a wrong
    class, ai, is closer than the margin, m, to at then it is penalized by the squared distance
    to the margin.

    By starting with a small margin of 0.2 and linearly increasing it during training to 0.9, we
    avoid dead capsules in the earlier layers.
    """
    def __init__(self, num_classes, reconstruction,
                 delta=0.1, margin=0.2, max_margin=0.9):
        """

        Args:
            num_classes: number of classes being predicted
            margin: scalar, the margin after subtracting 0.5 from raw_logits.
            downweight: scalar, the factor for negative cost.
            reconstruction: whether to include reconstruction loss
        """
        super().__init__()

        if num_classes < 2:
            raise ValueError("Number of classes must be greater than 1.")
        if delta < 0:
            raise ValueError("Margin annealing step size (delta) cannot be negative.")
        if margin > max_margin:
            raise ValueError("Max margin value must be greater than or equal to"
                             " the initial margin.")

        self.num_classes = num_classes
        self.delta = delta       # Amount by which to linearly anneal the margin by during training
        self.margin = margin
        self.max_margin = max_margin

        self.reconstruction = reconstruction
        if reconstruction:
            self.reconstruction_loss = ReconstructionLoss()

    def increment_margin(self):
        """
        Linearly anneal the margin during training by step size delta.
        """
        new_margin = self.margin + self.delta
        self.margin = min(new_margin, self.max_margin)

    def forward(self, logits, labels, reconstruction=None, image=None):
        """
        Args:
            logits: tensor, output of the model
            labels: tensor, labeled samples
            image: tensor, the reconstruction target image.
            reconstruction: tensor, the reconstruction image.

        Returns:
            loss: total loss
        """
        # margin = self.m_min + (self.m_max - self.m_min) * r
        # one-hot encoding of ground truth

        if self.reconstruction:
            if image is None:
                raise ValueError("Input image needed to compute reconstruction loss.")
            if reconstruction is None:
                raise ValueError("Reconstructed image needed to compute reconstruction loss.")

        labels = one_hot(labels, self.num_classes).float()

        logits = logits.view(-1, 1, self.num_classes)
        labels = labels.unsqueeze(dim=2)
        target = logits @ labels

        # (max(0, m - (at - ai)))^2
        classification_loss = F.relu(self.margin - (target - logits)) ** 2
        classification_loss = classification_loss @ (1 - labels)
        classification_loss = classification_loss.mean()

        all_loss = classification_loss
        reconstruction_loss = None
        if self.reconstruction:
            reconstruction_loss = self.reconstruction_loss(image, reconstruction)
            all_loss += reconstruction_loss

        return all_loss, classification_loss, reconstruction_loss


class MarginLoss(_Loss):

    def __init__(self, num_classes, reconstruction,
                 margin=0.4, downweight=0.5):
        """
        Penalizes deviations from margin for each logit.
        Args:
            num_classes: number of classes being predicted
            margin: scalar, the margin after subtracting 0.5 from raw_logits.
            downweight: scalar, the factor for negative cost.
            reconstruction: whether to include reconstruction loss
        """
        super().__init__()

        if num_classes < 2:
            raise ValueError("Number of classes must be greater than 1.")

        self.num_classes = num_classes
        self.margin = margin
        self.downweight = downweight
        self.reconstruction = reconstruction
        if reconstruction:
            self.reconstruction_loss = ReconstructionLoss()

    def forward(self, raw_logits, labels, reconstruction=None, image=None):
        """
        Each wrong logit costs its distance to margin. For negative logits margin is
        0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
        margin is 0.4 from each side.

        Args:
            raw_logits: tensor, model predictions in range [0, 1]
            labels: tensor, ground truth
            image: The reconstruction target image.
            reconstruction: reconstructed image

        Returns:
            A tensor with the mean cost for each data point of shape [batch_size].
        """
        # one-hot encoding of ground truth
        if self.reconstruction:
            if image is None:
                raise ValueError("Input image needed to compute reconstruction loss.")
            if reconstruction is None:
                raise ValueError("Reconstructed image needed to compute reconstruction loss.")

        labels = one_hot(labels, self.num_classes).float()
        logits = raw_logits - 0.5
        positive_cost = labels * (logits < self.margin).float() * \
                        (logits - self.margin) ** 2
        negative_cost = (1 - labels) * (logits > -self.margin).float() * \
                        (logits + self.margin) ** 2
        classification_loss = 0.5 * positive_cost + self.downweight * 0.5 * negative_cost
        classification_loss = classification_loss.sum(dim=1).mean()

        all_loss = classification_loss
        reconstruction_loss = None
        if self.reconstruction:
            reconstruction_loss = self.reconstruction_loss(image, reconstruction)
            all_loss += reconstruction_loss

        return all_loss, classification_loss, reconstruction_loss


class ReconstructionLoss(_Loss):
    """
    Calculate the loss between the target image and the output of the
    decoder sub-network. This is just the per-pixel euclidean distance.
    """
    def __init__(self, balance_factor=0.0005):
        """
        Args:
            balance_factor: scalar, downweight the loss to be in valid range
        """
        super().__init__()
        self.balance_factor = balance_factor    # factor to scale the loss down by so it doesn't dominate

    def forward(self, image, reconstruction):
        """
        Args:
            image: The reconstruction target image.
            reconstruction: The reconstruction image.

        Returns:
            reconstruction loss, the mean squared difference
            between the reconstruction and target image
        """
        image_2d = image.view(image.size(0), -1)
        distance = (image_2d - reconstruction) ** 2
        loss = distance.sum(-1)
        batch_loss = loss.mean()
        balanced_loss = self.balance_factor * batch_loss
        return balanced_loss
