# Use this code for Part (h) to fix the error calculation:

# Complete predict function in Logistic.py file and compute the percentage of mis-classified points
y_pred = logistic.predict(X_test)

# Ensure both arrays are flattened for proper comparison
y_test_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()

# Calculate error percentage
test_err = np.mean(y_test_flat != y_pred_flat) * 100

print(f"Test error: {test_err:.2f}%")
print(f"y_test shape: {y_test.shape}, flattened shape: {y_test_flat.shape}")
print(f"y_pred shape: {y_pred.shape}, flattened shape: {y_pred_flat.shape}")
print(f"Unique values in y_test: {np.unique(y_test)}")
print(f"Unique values in y_pred: {np.unique(y_pred)}")

# If you want to see class distribution
print(f"Number of class 1 in test set: {np.sum(y_test_flat == 1)}")
print(f"Number of class -1 in test set: {np.sum(y_test_flat == -1)}")
print(f"Number of class 1 in predictions: {np.sum(y_pred_flat == 1)}")
print(f"Number of class -1 in predictions: {np.sum(y_pred_flat == -1)}") 