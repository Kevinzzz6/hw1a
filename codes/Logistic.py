import numpy as np

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
        self.reg  = reg_param
        self.dim = [d+1, 1]
        self.w = np.zeros(self.dim)
    def gen_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,d) containing the data.
        Returns:
         - X_out an augmented training data to a feature vector e.g. [1, X].
        """
        N,d = X.shape
        X_out= np.zeros((N,d+1))
        # ================================================================ #
        # YOUR CODE HERE:
        # IMPLEMENT THE MATRIX X_out=[1, X]
        # ================================================================ #
        X_out[:, 0] = 1
        X_out[:, 1:] = X
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return X_out
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 labels 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w) 
        N, d = X.shape 
        
        # ================================================================ #
        # YOUR CODE HERE:
        # Calculate the loss function of the logistic regression
        # save loss function in loss
        # Calculate the gradient and save it as grad
        # ================================================================ #
        X_features = self.gen_features(X)
        
        # Ensure y is flattened
        y = y.reshape(-1)
        
        # Convert y from {-1,1} to {0,1} for binary cross-entropy calculation
        y_01 = (y + 1) / 2
        
        # Calculate scores and apply sigmoid with numerical stability
        scores = X_features.dot(self.w)
        # Clip scores to avoid overflow
        scores_clipped = np.clip(scores, -20, 20)
        h_w = 1.0 / (1.0 + np.exp(-scores_clipped))
        
        # Calculate binary cross-entropy loss with numerical stability
        epsilon = 1e-15  # Small value to avoid log(0)
        h_w_safe = np.clip(h_w, epsilon, 1.0 - epsilon)
        
        # Binary cross-entropy loss
        loss = -np.mean(y_01 * np.log(h_w_safe) + (1 - y_01) * np.log(1 - h_w_safe))
        
        # Add regularization if needed
        if self.reg > 0:
            loss += (self.reg / 2) * np.sum(self.w[1:] ** 2)  # Don't regularize the bias term
        
        # Calculate the gradient (in original labels format)
        error = h_w - y_01.reshape(-1, 1)
        grad = (1.0/N) * X_features.T.dot(error)
        
        # Add regularization to gradient if needed
        if self.reg > 0:
            reg_grad = np.zeros_like(self.w)
            reg_grad[1:] = self.reg * self.w[1:]  # Don't regularize the bias term
            grad += reg_grad
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return loss, grad
    
    def train_LR(self, X, y, eta=1e-3, batch_size=1, num_iters=1000) :
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
        loss_history = []
        N,d = X.shape
        
        # Reset weights
        self.w = np.zeros(self.dim)
        
        # Make sure y is the right shape and format (-1 or 1)
        y = y.flatten() if hasattr(y, 'flatten') else y
        
        for t in range(num_iters):
            # Sample batch
            indices = np.random.choice(N, batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            
            # Compute loss and gradient
            loss, grad = self.loss_and_grad(X_batch, y_batch)
            
            # Update weights using gradient descent
            self.w -= eta * grad
            
            # Track loss
            loss_history.append(loss)
            
            # Print progress occasionally
            if t % 1000 == 0:
                print(f"Iteration {t}, Loss: {loss:.6f}")
                
        return loss_history, self.w
    
    def predict(self, X):
        """
        Inputs:
        - X: N x d array of training data.
        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        # ================================================================ #
        # YOUR CODE HERE:
        # PREDICT THE LABELS OF X 
        # ================================================================ #
        X_features = self.gen_features(X)
        scores = X_features.dot(self.w)
        
        # Use a numerically stable sigmoid function
        prob = 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))
        
        # Convert to the required format (1 for dress, -1 for shirt)
        y_pred = 2 * (prob >= 0.5).astype(int) - 1
        
        # Ensure y_pred is a flattened array
        y_pred = y_pred.flatten()
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return y_pred