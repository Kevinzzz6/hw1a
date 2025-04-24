## PART (d) (Different Learning Rates):
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt

def test_learning_rates(X_train, y_train, X_test, y_test):
    """
    Tests different learning rates for linear regression
    and calculates test error for each rate
    """
    lrs = [1e-1, 1e-2, 1e-3, 1e-4]
    test_err = np.zeros((len(lrs), 1))
    # ================================================================ #
    # YOUR CODE HERE:
    # Train linear regression for different learning rates and average the test error over 10 trials
    # ================================================================ #
    from codes.Regression import Regression
    num_runs = 10
    batch_size = 30
    num_iters = 10000
    
    for lr_idx, eta in enumerate(lrs):
        run_errors = []
        
        # Run multiple trials to get average performance
        for run in range(num_runs):
            # Initialize regression model
            regression = Regression(m=1, reg_param=0)
            
            # Train with current learning rate
            loss_history, w = regression.train_LR(X_train, y_train, eta=eta, 
                                                 batch_size=batch_size, 
                                                 num_iters=num_iters)
            
            # Evaluate on test set
            y_pred = regression.predict(X_test)
            mse = np.mean((y_pred - y_test) ** 2)
            run_errors.append(mse)
        
        # Average the errors across runs
        test_err[lr_idx] = np.mean(run_errors)
        print(f"Learning rate {eta}: Average test MSE = {test_err[lr_idx][0]:.6f}")
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    # Plot results
    fig = plt.figure()
    plt.semilogx(lrs, test_err, 'o-')  # Use semilog scale for better visualization
    plt.xlabel('Learning Rate')
    plt.ylabel('Test MSE')
    plt.title('Effect of Learning Rate on Test Error')
    plt.grid(True)
    plt.show()
    fig.savefig('./plots/LR_learning_rates_test.pdf')
    
    return test_err

# To use this in the notebook, call:
# test_err = test_learning_rates(X_train, y_train, X_test, y_test) 