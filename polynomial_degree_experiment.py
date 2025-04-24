## PART (g) and PART (i): 
# Test different polynomial degrees for Regression

import numpy as np
import matplotlib.pyplot as plt
from codes.Regression import Regression

def polynomial_degree_experiment(X_train, y_train, X_test, y_test, use_regularization=False):
    """
    Tests different polynomial degrees for regression and plots 
    training and test losses
    
    Parameters:
    -----------
    X_train, y_train: Training data and labels
    X_test, y_test: Test data and labels
    use_regularization: Whether to use regularization (Part i)
    
    Returns:
    --------
    train_loss, test_loss: Arrays of training and test losses
    """
    max_degree = 10
    train_loss = np.zeros((max_degree, 1))
    test_loss = np.zeros((max_degree, 1))
    
    # ================================================================ #
    # YOUR CODE HERE:
    # Complete the following code to plot both the training and test loss 
    # for polynomial degrees from 1 to 10
    # ================================================================ #
    
    # Set regularization parameter based on the experiment
    reg_param = 0.1 if use_regularization else 0
    
    for m in range(1, max_degree + 1):
        # Initialize regression model with polynomial degree m
        regression = Regression(m=m, reg_param=reg_param)
        
        # Directly solve using the closed-form solution for efficiency
        train_loss[m-1], w = regression.closed_form(X_train, y_train)
        
        # Calculate test error
        y_pred = regression.predict(X_test)
        error = y_pred - y_test
        test_loss[m-1] = (1.0/(2*len(y_test))) * np.sum(error**2)
        
        # Add regularization to test loss if needed
        if reg_param > 0:
            test_loss[m-1] += (reg_param / 2) * np.sum(w[1:] ** 2)
        
        print(f"Degree {m}: Train loss = {train_loss[m-1][0]:.6f}, Test loss = {test_loss[m-1][0]:.6f}")
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    x_axis = np.arange(1, max_degree+1)
    plt.plot(x_axis, train_loss, 'bo-', label='Training Loss')
    plt.plot(x_axis, test_loss, 'ro-', label='Test Loss')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Loss')
    plt.title(f'Regression with {"Regularization" if use_regularization else "No Regularization"}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    if use_regularization:
        plt.savefig('./plots/polynomial_degree_with_reg.pdf')
    else:
        plt.savefig('./plots/polynomial_degree_no_reg.pdf')
    
    return train_loss, test_loss

# For Part (g) - No regularization:
# train_loss_g, test_loss_g = polynomial_degree_experiment(X_train, y_train, X_test, y_test, use_regularization=False)

# For Part (i) - With regularization:
# train_loss_i, test_loss_i = polynomial_degree_experiment(X_train, y_train, X_test, y_test, use_regularization=True) 