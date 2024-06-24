import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader
import numpy as np #remove
from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.
        #weights.shape = [features, classes]
        self.weights = torch.normal(0, weight_std, size=[n_features, n_classes])
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = torch.mm(x,self.weights)
        y_pred = torch.argmax(class_scores, axis=1)
        # ========================
        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        correct = np.where(y == y_pred)[0]
        acc = len(correct) / len(y)
        # ========================

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======

            #eval on train:
            for (batch, label) in iter(dl_train):
                    y_pred, class_scores = self.predict(batch)
                    loss = loss_fn(batch, label, class_scores, y_pred)
                    acc = self.evaluate_accuracy(label, y_pred)
                    train_res.accuracy.append(acc)
                    train_res.loss.append(loss / len(batch))
                    average_loss += loss
                    total_correct += acc * len(batch)

                    #update weights:
                    
                    #regularization loss
                    self.weights -= learn_rate * weight_decay * self.weights
                    
                    #normal loss
                    self.weights -= learn_rate * loss_fn.grad()

            #eval on valid
            for (batch, label) in iter(dl_train):
                y_pred, class_scores = self.predict(batch)
                loss = loss_fn(batch, label, class_scores, y_pred)
                acc = self.evaluate_accuracy(label, y_pred)
                valid_res.accuracy.append(acc)
                valid_res.loss.append(loss / len(batch))
                average_loss += loss
                total_correct += acc * len(batch)
            
            # ========================
            print(".", end="")
        
        average_loss = average_loss / (len(dl_train.dataset) + len(dl_valid.dataset))
        total_correct = total_correct / (len(dl_train.dataset) + len(dl_valid.dataset))
        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        if(has_bias):
            w_images = self.weights[1:, :].T.reshape((self.weights.shape[1], *img_shape)) #Ignore row 0 == bias
        else:
            w_images = self.weights.T.reshape((self.weights.shape[1], *img_shape))
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0, learn_rate=0, weight_decay=0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp = dict(weight_std=0.03, learn_rate=0.03, weight_decay=0.0003) #weight_std=0.03, learn_rate=0.09, weight_decay=0.001
    #raise NotImplementedError()
    # ========================

    return hp
