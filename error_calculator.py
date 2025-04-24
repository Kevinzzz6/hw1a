import numpy as np

def calculate_error_rate(y_true, y_pred):
    """
    Calculate the error rate (percentage of misclassifications) between 
    the predicted labels and ground truth labels.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels (can be 1D or 2D array)
    y_pred : array-like
        Predicted labels (can be 1D or 2D array)
    
    Returns:
    --------
    error_rate : float
        Percentage of misclassifications (0-100)
    """
    # Ensure both arrays are 1D for proper comparison
    y_true_flat = y_true.flatten() if hasattr(y_true, 'flatten') else np.array(y_true).flatten()
    y_pred_flat = y_pred.flatten() if hasattr(y_pred, 'flatten') else np.array(y_pred).flatten()
    
    # Verify they have the same length
    if len(y_true_flat) != len(y_pred_flat):
        raise ValueError(f"Length mismatch: y_true has {len(y_true_flat)} elements, y_pred has {len(y_pred_flat)} elements")
    
    # Calculate error rate as percentage
    error_rate = np.mean(y_true_flat != y_pred_flat) * 100
    
    return error_rate

def debug_predictions(y_true, y_pred, name=""):
    """
    Print detailed debug information about predictions and actual values
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    name : str
        Optional name to include in the debug output
    """
    y_true_flat = y_true.flatten() if hasattr(y_true, 'flatten') else np.array(y_true).flatten()
    y_pred_flat = y_pred.flatten() if hasattr(y_pred, 'flatten') else np.array(y_pred).flatten()
    
    error_rate = calculate_error_rate(y_true, y_pred)
    
    print(f"{name} Debug Information:")
    print(f"  y_true shape: {np.array(y_true).shape}, flattened: {y_true_flat.shape}")
    print(f"  y_pred shape: {np.array(y_pred).shape}, flattened: {y_pred_flat.shape}")
    print(f"  y_true unique values: {np.unique(y_true)}")
    print(f"  y_pred unique values: {np.unique(y_pred)}")
    print(f"  Error rate: {error_rate:.2f}%")
    print(f"  Correctly classified: {100 - error_rate:.2f}%")
    
    # Show distribution of predictions
    if len(np.unique(y_true)) <= 10:  # Only show confusion for small number of classes
        print(f"  Value counts in true labels: {np.bincount(y_true_flat + 1 if -1 in np.unique(y_true) else y_true_flat)}")
        print(f"  Value counts in predictions: {np.bincount(y_pred_flat + 1 if -1 in np.unique(y_pred) else y_pred_flat)}")
        
    return error_rate 