from cProfile import label
from cmath import cos
import copy 
import math
import numpy as np
import matplotlib.pyplot as plt

# reduce display precision on numpy arrays
np.set_printoptions(precision=2)

def compute_cost(X,y, w, b):    

    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2 *m)
    return cost

def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 

    """
    m,n = X.shape   #(number of examples, number of features)
    dj_dw = np.zeros((n,)) 
    dj_db = 0.
    for i in range(m):
        err =  (np.dot(X[i], w) +b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err *X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw, dj_db


def gradient_descent (X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 

    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in) # avoid to modifying global w_in
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(X,y,w,b)

        # Update parameters 
        w =  w - alpha * dj_dw
        b =  b - alpha * dj_db

        # Save cost J after each iteration
        if i < 10000:
            J_history.append( cost_function (X, y, w, b))
         # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ")
 
    return w, b, J_history #return w and J,w history for graphing


if __name__ ==  "__main__":

    # Some Gradient descent setting
    iteration = 10000
    tmp_alpha = 5.0e-7

    # X_train is input variable (size of house * 1000)
    # y_train is output price is dollar
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])

    # data is stored in numpy array/matrix
    print(f"x Shape: {X_train.shape}, X Type:{type(X_train)})")
    print(X_train)
    print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
    print(y_train)

    # initialize parameters
    b_init = 785.1811367994083
    w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
    print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

    # Compute and display cost using our pre-chosen optimal parameters. 
    cost = compute_cost(X_train, y_train, w_init, b_init)
    print(f'Cost at optimal w : {cost}')


    #number of element
    m = X_train.shape[0]  # Or we can use len(x_train)
    print(f"number of element is = {m}")

    # run gradient descent 
    w_final, b_final, J_hist = gradient_descent(X_train, y_train, w_init, b_init,
                                                        compute_cost, compute_gradient, 
                                                        tmp_alpha, iteration)
    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
    m,_ = X_train.shape
    for i in range(m):
        print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

    # plot cost versus iteration  
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(J_hist)
    ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
    ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
    ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
    ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
    plt.show()

