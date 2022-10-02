from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    # W*X 한다음에 svm로스 구한후 합산하고 평균 냄
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                #스코어 부분 업데이트
                dW[:,j] += X[i]
                #해당 클래스 부분 업데이트
                dW[:,y[i]] += (-1) * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    # L2 regularization
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # hinge loss part ?
    
    # regularization part
    dW += 2*W*reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    num_classes = W.shape[1]
    num_train = X.shape[0]
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X@W
    correct_class_score = scores[np.arange(0,num_train),y]
    margin = scores - correct_class_score.reshape([-1,1]) + 1
    margin *= margin>0
    margin[np.arange(0,num_train),y] = 0
    loss = margin.sum()
    loss /= num_train
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    margin_b = (margin>0).astype(int)
    margin_sum = margin_b.sum(axis = 1)
    margin_b[np.arange(0,num_train),y] = -margin_sum

    X_new = X[:,np.newaxis]
    X_new = X_new.T * margin_b.T

    dW += X_new.sum(axis = 2)
    dW /= num_train
    dW += 2*W*reg
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

if __name__ == "__main__":
    #data parts
    from data_utils import load_CIFAR10
    from time import time
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    num_training = 49000
    num_dev = 500
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    X_train = X_train[mask]
    y_train = y_train[mask]
    X_train = np.reshape(X_train, (X_train.shape[0], -1))

    mean_image = np.mean(X_train, axis=0)
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    X_dev -= mean_image
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    W = np.random.randn(3073, 10) * 0.0001

    #train
    tic = time()
    loss, grad = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
    print(time() - tic)
    tic = time()
    loss_n, grad_n = svm_loss_naive(W, X_dev, y_dev, 0.000005)
    print(time() - tic)

    print(loss - loss_n)
    print((grad - grad_n).sum())

    