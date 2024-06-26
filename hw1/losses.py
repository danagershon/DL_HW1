import abc
import torch
import numpy as np

class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        truth_scores_column = x_scores[:, y].diag().reshape([x_scores.shape[0],1]) #On the diagonal, it has [i, y[i]]
        M = self.delta + x_scores - truth_scores_column.repeat(1, x_scores.shape[-1]) #[N, C]

        loss = torch.sum(torch.maximum(M, torch.zeros(M.shape)), dim=1) - self.delta #Sum on columns. Remove delta for the column where j==y_i
        loss = torch.mean(loss)
        
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx["m"] = M
        self.grad_ctx["y"] = y
        self.grad_ctx["X"] = x
        self.grad_ctx["C"] = x_scores.shape[1]
        # ========================
        
        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        G = (self.grad_ctx["m"] > 0).float()
        G_row_penalty = -1 * torch.sum(G, dim=1).reshape([self.grad_ctx["m"].shape[0], 1])

        G_truth_mask = torch.nn.functional.one_hot(self.grad_ctx["y"], num_classes=self.grad_ctx["C"]) #G_mask[i,j] = 1 iff y[i] = j
        
        G_truth_penalty = torch.mul(G_row_penalty, G_truth_mask)
        G += G_truth_penalty #Fix G where j==y_i, by removing the extra 1 there and turning it to -1's (sum over all row)
        
        grad = torch.matmul(self.grad_ctx["X"].T, G) * 1 / self.grad_ctx["X"].shape[0] #Average by N
        
        return grad
