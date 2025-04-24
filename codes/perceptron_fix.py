import numpy as np
import matplotlib.pyplot as plt

def train_perceptron(X_train, y_train, X_test, y_test, max_iterations=1000, eval_freq=10):
    """
    Optimized Perceptron implementation for binary classification.
    
    Parameters:
    -----------
    X_train: Training data of shape (N, d)
    y_train: Training labels of shape (N, 1)
    X_test: Test data of shape (N_test, d)
    y_test: Test labels of shape (N_test, 1)
    max_iterations: Maximum number of iterations
    eval_freq: Frequency to evaluate and log errors
    
    Returns:
    --------
    train_errors: List of training errors
    test_errors: List of test errors
    loss_hist: List of number of misclassified training points
    squared_norms: List of squared L2 norms of the weight vector
    W: Final weight vector
    iter_points: Iteration points for plotting
    """
    # Get dimensions
    N = X_train.shape[0]
    N_test = X_test.shape[0]
    d = X_train.shape[1]
    
    # Add bias term
    X_train_h = np.hstack((np.ones((N, 1)), X_train))
    X_test_h = np.hstack((np.ones((N_test, 1)), X_test))
    
    # Initialize weights and tracking variables
    W = np.zeros((d+1, 1))
    train_errors = []
    test_errors = []
    loss_hist = []
    squared_norms = []
    
    # Flatten y to make comparisons easier
    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()
    
    for iter in range(max_iterations):
        # Flag to track if we found a misclassified point in this iteration
        found_misclassified = False
        
        # Randomize the order of training examples
        indices = np.random.permutation(N)
        
        # Go through dataset in random order
        for i in indices:
            x_i = X_train_h[i].reshape(-1, 1)  # Make sure it's a column vector
            y_i = y_train_flat[i]
            
            # Check if this point is misclassified
            score = np.dot(W.T, x_i).item()
            pred = 1 if score > 0 else -1
            
            if pred != y_i:
                # Update weights using this misclassified point
                W = W + y_i * x_i
                found_misclassified = True
                break  # Move to next iteration after one update
        
        # Evaluate error metrics only at specified intervals
        if iter % eval_freq == 0 or iter == max_iterations - 1 or not found_misclassified:
            # Training predictions and error
            scores_train = X_train_h.dot(W)
            y_pred_train = np.sign(scores_train)
            y_pred_train[y_pred_train == 0] = -1  # Handle zero case
            
            # Count misclassified points
            train_misclassified = np.sum(y_pred_train.flatten() != y_train_flat)
            train_error = train_misclassified / N * 100
            train_errors.append(train_error)
            
            # Test predictions and error
            scores_test = X_test_h.dot(W)
            y_pred_test = np.sign(scores_test)
            y_pred_test[y_pred_test == 0] = -1  # Handle zero case
            test_misclassified = np.sum(y_pred_test.flatten() != y_test_flat)
            test_error = test_misclassified / N_test * 100
            test_errors.append(test_error)
            
            # Other statistics
            squared_norm = np.linalg.norm(W)**2
            squared_norms.append(squared_norm)
            loss_hist.append(train_misclassified)
            
            # If no misclassifications or we didn't find any to update, we've converged
            if train_misclassified == 0 or not found_misclassified:
                print(f"Converged after {iter} iterations!")
                break
    
    # Create iteration points for plotting (based on eval_freq)
    iter_points = np.arange(0, len(train_errors)) * eval_freq
    
    # Ensure the last point corresponds to the final iteration if not divisible by eval_freq
    if (iter % eval_freq) != 0 and iter < max_iterations - 1:
        iter_points[-1] = iter
        
    return train_errors, test_errors, loss_hist, squared_norms, W, iter_points

def plot_perceptron_results(iter_points, train_errors, test_errors, loss_hist):
    """
    Plot the results of Perceptron training.
    """
    plt.figure(figsize=(14, 5))
    
    # Plot training and test errors
    plt.subplot(1, 2, 1)
    plt.plot(iter_points, train_errors, 'b-', label='Training Error (%)')
    plt.plot(iter_points, test_errors, 'r-', label='Test Error (%)')
    plt.xlabel('Iterations')
    plt.ylabel('Error (%)')
    plt.legend()
    plt.title('Perceptron Training and Test Errors')
    
    # Plot loss history
    plt.subplot(1, 2, 2)
    plt.plot(iter_points, loss_hist, 'g-', label='Loss (# misclassified)')
    plt.xlabel('Iterations')
    plt.ylabel('Number of Misclassified Points')
    plt.title('Perceptron Loss History')
    plt.tight_layout()
    
    # Display the plot
    plt.show() 