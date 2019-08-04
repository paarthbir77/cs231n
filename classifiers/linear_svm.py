import numpy as np


def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    y = [int(x) for x in y]
    correct_class_scores = scores[np.arange(num_train), y].reshape(num_train, 1)
    margin = np.maximum(0, scores - correct_class_scores + 1)
    margin[np.arange(num_train), y] = 0  # do not consider correct class in loss
    loss = margin.sum() / num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    margin[margin > 0] = 1
    valid_margin = margin.sum(axis=1)
    margin[np.arange(num_train), y] -= valid_margin
    dW = (X.T).dot(margin) / num_train
    # dW /= num_train
    # Regularization gradient
    dW = dW + reg * 2 * W
    return loss, dW
