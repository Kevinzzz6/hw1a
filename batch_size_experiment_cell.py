## PART (i): 
Batch = [1, 50, 100, 200, 300]
test_err = np.zeros((len(Batch), 1))
# ================================================================ #
# YOUR CODE HERE:
# Train the Logistic regression for different batch size. Average the test error over 10 times
# ================================================================ #
num_runs = 10
eta = 1e-5
num_iters = 5000

# Ensure y values are properly formatted
y_train_formatted = y_train.flatten() if hasattr(y_train, 'flatten') else y_train
y_test_formatted = y_test.flatten() if hasattr(y_test, 'flatten') else y_test

for b_idx, batch_size in enumerate(Batch):
    batch_errors = []
    
    # Run 10 times and average the results
    for run in range(num_runs):
        # Initialize a new logistic regression model
        logistic = Logistic(d=X_train.shape[1], reg_param=0)
        
        # Train with the current batch size
        loss_history, w = logistic.train_LR(X_train, y_train_formatted, eta=eta, batch_size=batch_size, num_iters=num_iters)
        
        # Compute test error (percentage of misclassified points)
        y_pred = logistic.predict(X_test)
        error = np.mean(y_pred != y_test_formatted) * 100  # Convert to percentage
        batch_errors.append(error)
    
    # Average the errors for this batch size
    test_err[b_idx] = np.mean(batch_errors)
    print(f"Batch size {batch_size}: Average test error = {test_err[b_idx][0]:.2f}%")
# ================================================================ #
# END YOUR CODE HERE
# ================================================================ # 
fig = plt.figure()
plt.plot(Batch, test_err)
plt.xlabel('Batch_size')
plt.ylabel('Test_error (%)')
plt.title('Effect of Batch Size on Test Error')
plt.grid(True)
plt.show()
fig.savefig('./plots/LR_Batch_test.pdf') 