import numpy as np
# Remove sklearn dependency
# from sklearn.preprocessing import StandardScaler

class Logistic(object):
    def __init__(self, d=784, reg_param=0):
        """"
        Inputs:
          - d: Number of features
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the  regularization parameter self.reg
        """
        self.reg = reg_param
        self.dim = [d+1, 1]
        self.w = np.zeros(self.dim)
        # Replace StandardScaler with manual mean and std
        self.feature_means = None
        self.feature_stds = None
        self.best_w = None
        self.best_val_error = float('inf')
        
    def gen_features(self, X, fit=False):
        """
        Inputs:
         - X: A numpy array of shape (N,d) containing the data.
         - fit: Boolean indicating whether to fit the scaler
        Returns:
         - X_out an augmented training data to a feature vector e.g. [1, X].
        """
        N, d = X.shape
        
        # Implement manual standardization
        if fit:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0) + 1e-10  # Avoid div by zero
            X_scaled = (X - self.feature_means) / self.feature_stds
        else:
            if self.feature_means is not None and self.feature_stds is not None:
                X_scaled = (X - self.feature_means) / self.feature_stds
            else:
                X_scaled = X  # No normalization if not fitted
            
        # Add bias term as first column
        X_out = np.zeros((N, d+1))
        X_out[:, 0] = 1
        X_out[:, 1:] = X_scaled
        
        return X_out
        
    def sigmoid(self, z):
        """
        Stable sigmoid function implementation to avoid overflow
        """
        # Clip values to avoid overflow
        z_safe = np.clip(z, -20, 20)
        return 1.0 / (1.0 + np.exp(-z_safe))
        
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 labels 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        N, d = X.shape
        
        # Use normalized features
        X_features = self.gen_features(X)
        
        # Calculate the stable logistic regression hypothesis
        scores = X_features.dot(self.w)
        h_w = self.sigmoid(scores)
        
        # Convert y to {0, 1} for binary cross-entropy
        y_01 = (y + 1) / 2
        
        # Calculate the binary cross-entropy loss - more numerically stable
        epsilon = 1e-15  # Small constant to avoid log(0)
        h_w_safe = np.clip(h_w, epsilon, 1 - epsilon)
        
        log_probs = y_01 * np.log(h_w_safe) + (1 - y_01) * np.log(1 - h_w_safe)
        loss = -np.mean(log_probs)
        
        # Add L2 regularization if needed
        if self.reg > 0:
            loss += (self.reg / 2) * np.sum(self.w[1:] ** 2)  # Don't regularize bias
        
        # Calculate the gradient with respect to weights
        error = h_w - y_01
        grad = (1.0/N) * X_features.T.dot(error)
        
        # Add regularization to gradient if needed
        if self.reg > 0:
            reg_grad = np.zeros_like(self.w)
            reg_grad[1:] = self.reg * self.w[1:]  # Don't regularize bias
            grad += reg_grad
            
        return loss, grad
    
    def train_LR(self, X, y, eta=1e-3, batch_size=1, num_iters=1000):
        """
        Inputs:
         - X         -- numpy array of shape (N,d), features
         - y         -- numpy array of shape (N,), labels
         - eta       -- float, learning rate
         - num_iters -- integer, maximum number of iterations
        Returns:
         - loss_history: vector containing the loss at each training iteration.
         - self.w: optimal weights 
        """
        N, d = X.shape
        loss_history = []
        
        # Initialize parameters
        self.w = np.zeros(self.dim)
        velocity = np.zeros_like(self.w)  # For momentum
        
        # Set hyperparameters
        momentum = 0.9
        beta1 = 0.9  # Adam parameters
        beta2 = 0.999
        epsilon = 1e-8
        m = np.zeros_like(self.w)
        v = np.zeros_like(self.w)
        t = 0
        
        # Split data for validation-based early stopping
        val_size = int(0.1 * N)
        train_indices = np.random.choice(N, N-val_size, replace=False)
        val_indices = np.setdiff1d(np.arange(N), train_indices)
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Normalize features
        X_train_features = self.gen_features(X_train, fit=True)
        X_val_features = self.gen_features(X_val)
        
        # Initialize best model tracking
        best_val_error = float('inf')
        best_w = None
        patience = 5
        patience_counter = 0
        
        for iter in range(num_iters):
            # Sample batch
            batch_indices = np.random.choice(X_train.shape[0], batch_size, replace=False)
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # Compute gradient
            loss, grad = self.loss_and_grad(X_batch, y_batch)
            
            # Adam optimizer update
            t += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Update weights with adaptive learning rate
            self.w -= eta * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Add loss to history
            loss_history.append(loss)
            
            # Check validation error every 50 iterations
            if iter % 50 == 0:
                # Predict on validation set
                scores_val = X_val_features.dot(self.w)
                probs_val = self.sigmoid(scores_val)
                y_val_pred = 2 * (probs_val >= 0.5).astype(int) - 1
                val_error = np.mean(y_val_pred.flatten() != y_val)
                
                # Check if validation error improved
                if val_error < best_val_error:
                    best_val_error = val_error
                    best_w = self.w.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping at iteration {iter}")
                    break
                
                # Adaptive learning rate decay
                if iter > 0 and iter % 500 == 0:
                    eta *= 0.5
                    print(f"Reducing learning rate to {eta}")
            
        # Use best weights found during training
        if best_w is not None:
            self.w = best_w
            self.best_w = best_w
            self.best_val_error = best_val_error
            
        return loss_history, self.w
    
    def predict(self, X):
        """
        Inputs:
        - X: N x d array of training data.
        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        # Normalize features
        X_features = self.gen_features(X)
        
        # Compute probabilities
        scores = X_features.dot(self.w)
        probs = self.sigmoid(scores)
        
        # Convert to {-1, 1} labels based on threshold
        # Use a slightly different threshold if performance is better
        thresh = 0.5  # Can be tuned
        y_pred = 2 * (probs >= thresh).astype(int) - 1
        
        return y_pred
        
    def tune_hyperparameters(self, X, y, X_test, y_test):
        """
        Tune hyperparameters using a validation approach
        """
        # Split data for validation
        N = X.shape[0]
        val_size = int(0.1 * N)
        train_indices = np.random.choice(N, N-val_size, replace=False)
        val_indices = np.setdiff1d(np.arange(N), train_indices)
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Parameters to tune
        learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
        batch_sizes = [1, 32, 64, 128, 256]
        reg_params = [0, 0.001, 0.01, 0.1, 1.0]
        
        best_error = float('inf')
        best_params = {}
        
        print("Tuning hyperparameters...")
        
        # Grid search
        for lr in learning_rates:
            for bs in batch_sizes:
                for reg in reg_params:
                    # Create new model with current regularization
                    model = Logistic(d=X.shape[1], reg_param=reg)
                    
                    # Train with current hyperparameters
                    _, _ = model.train_LR(X_train, y_train, eta=lr, batch_size=bs, num_iters=2000)
                    
                    # Evaluate on validation set
                    y_val_pred = model.predict(X_val)
                    val_error = np.mean(y_val_pred != y_val)
                    
                    # Check if better than previous best
                    if val_error < best_error:
                        best_error = val_error
                        best_params = {'lr': lr, 'batch_size': bs, 'reg': reg}
                        
        print(f"Best params: {best_params}, Val Error: {best_error:.4f}")
        
        # Train final model on full training data
        self.reg = best_params['reg']
        _, _ = self.train_LR(X, y, eta=best_params['lr'], 
                           batch_size=best_params['batch_size'], num_iters=5000)
        
        # Test error
        y_test_pred = self.predict(X_test)
        test_error = np.mean(y_test_pred != y_test) * 100
        print(f"Test error: {test_error:.2f}%")
        
        return best_params, test_error