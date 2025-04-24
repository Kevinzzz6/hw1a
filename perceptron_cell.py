# PART (b),(c): 
# Implement the perceptron Algorithm and compute the number of mis-classified point
from codes.perceptron_fix import train_perceptron, plot_perceptron_results

N = X_train.shape[0] # Number of data point train
N_test = X_test.shape[0] # Number of data point test
d = X_train.shape[1] # Number of features

# ================================================================ #
# YOUR CODE HERE:
# complete the following code to plot both the training and test accuracy in the same plot
# for m range from 1 to N
# ================================================================ #

# Ensure y values are properly formatted for the perceptron
# Convert to numpy arrays and ensure they are the right shape
y_train_formatted = y_train.flatten() if hasattr(y_train, 'flatten') else y_train
y_test_formatted = y_test.flatten() if hasattr(y_test, 'flatten') else y_test

# Train the perceptron - using the optimized implementation
train_errors, test_errors, loss_hist, squared_norms, W, iter_points = train_perceptron(
    X_train, y_train_formatted, X_test, y_test_formatted, 
    max_iterations=1000, 
    eval_freq=10
)

# Plot the results
plot_perceptron_results(iter_points, train_errors, test_errors, loss_hist)

# Print final results
print(f"Final training error: {train_errors[-1]:.2f}%")
print(f"Final test error: {test_errors[-1]:.2f}%")
print(f"Final squared L2 norm of weights: {squared_norms[-1]:.2f}")
print(f"Number of iterations: {iter_points[-1]+1}")
print(f"Convergence: {'Yes' if loss_hist[-1] == 0 else 'No'}")
# ================================================================ #
# END YOUR CODE HERE
# ================================================================ # 