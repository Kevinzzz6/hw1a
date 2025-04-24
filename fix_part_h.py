# PART (h)
# Complete predict function in Logisitc.py file and compute the percentage of mis-classified points
y_pred = logistic.predict(X_test)
# Flatten y_test to match y_pred's shape for proper comparison
y_test_flat = y_test.flatten() 
test_err = np.mean(y_test_flat != y_pred) * 100  # Calculate percentage using mean
print(f"Test error: {test_err:.2f}%")
print(f"y_test shape: {y_test.shape}, flattened: {y_test_flat.shape}, values: {np.unique(y_test)}")
print(f"y_pred shape: {y_pred.shape}, values: {np.unique(y_pred)}") 