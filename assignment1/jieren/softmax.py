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
    
    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        correct_class_score = exp_scores[y[i]]
        exp_scores_sum = np.sum(exp_scores)
        loss -= np.log(correct_class_score / exp_scores_sum)
        
        for j in range(num_classes):
            if j == y[i]:
                dW[:,j] -= X[i]*(exp_scores_sum-correct_class_score)/exp_scores_sum
            else:
                dW[:,j] += X[i]*exp_scores[j]/exp_scores_sum


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    dW /= num_train 
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X.dot(W)
    scores -= np.argmax(scores, axis=1)[:,np.newaxis]
    exp_scores = np.exp(scores)
    correct_class_score = exp_scores[np.arange(num_train),y]
    exp_scores_sum = np.sum(exp_scores, axis=1)
    
    loss = np.sum(-np.log(correct_class_score / exp_scores_sum))
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    exp_scores[np.arange(num_train),y] = correct_class_score - exp_scores_sum
    exp_scores /= exp_scores_sum[:,np.newaxis]
    dW = np.matmul(X.T, exp_scores)
    dW /= num_train 
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
