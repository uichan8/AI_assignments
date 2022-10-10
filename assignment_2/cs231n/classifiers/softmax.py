from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]

    #loss part
    for i in range(num_train):
      scores = X[i].dot(W)
      scores -= scores.max() 

      e_scores = np.exp(scores)
      e_sum = e_scores.sum()
      sm = e_scores[y[i]] / e_sum
      sm = np.clip(sm,1e-10,1) #미분 불가점 1

      ce = -np.log(sm)
      loss += ce

      #back
      dsm_dce = -1/sm
      des_dsm = np.zeros([num_classes])
      for j in range(num_classes):
        if j == y[i]:
          des_dsm[j] = (e_sum - e_scores[y[i]])/e_sum**2
        else:
          des_dsm[j] = -e_scores[y[i]]/e_sum**2
        ds_des = np.exp(scores)
        dW_ds = X[i].reshape(-1,1)

      dW += dW_ds * (ds_des * (des_dsm * dsm_dce))

    dW /= num_train
    loss /= num_train

    #reg part
    loss += reg * (W*W).sum()
    dW += 2*W*reg
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    score = X@W
    score -= score.max(axis = 1).reshape(-1,1)
           
    e_scores = np.exp(score)
    e_sum = e_scores.sum(axis = 1)

    sm = e_scores[np.arange(num_train),y] / e_sum
    sm = np.clip(sm,1e-10,1) #미분 불가점 1

    ce = -np.log(sm)
    loss += ce.sum()

    loss /= num_train
    loss += reg * (W*W).sum()

    #back
    dsm_dce = -1/sm
    des_dsm = 
    ds_des = 
    dW_ds = 
    
    


    

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
    loss_n, grad_n = softmax_loss_naive(W, X_dev, y_dev, 0.000005)
    loss_n, grad_n = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)

    # #train
    # tic = time()
    # loss, grad = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
    # print(time() - tic)
    # tic = time()
    # loss_n, grad_n = svm_loss_naive(W, X_dev, y_dev, 0.000005)
    # print(time() - tic)

    # print(loss - loss_n)
    # print((grad - grad_n).sum())
