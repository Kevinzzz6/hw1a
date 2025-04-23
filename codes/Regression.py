import numpy as np

class Regression(object):
    def __init__(self, m=1, reg_param=0):
        """"
        Inputs:
          - m Polynomial degree
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the polynomial degree self.m
         - Initialize the  regularization parameter self.reg
        """
        self.m = m
        self.reg  = reg_param
        self.dim = [m+1 , 1]
        self.w = np.zeros(self.dim)
    def gen_poly_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,1) containing the data.
        Returns:
         - X_out an augmented training data to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        """
        N,d = X.shape
        m = self.m
        X_out= np.zeros((N,m+1))
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X]
            # ================================================================ #
            X_out[:, 0] = 1
            X_out[:, 1] = X.flatten()
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            X_out[:, 0] = 1
            for i in range(1, m+1):
                X_out[:, i] = np.power(X.flatten(), i)
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return X_out  
    
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 targets 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w) 
        m = self.m
        N,d = X.shape 
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # Calculate the loss function of the linear regression
            # save loss function in loss
            # Calculate the gradient and save it as grad
            #
            # ================================================================ #
            X_features = self.gen_poly_features(X)
            
            # Calculate predictions
            h_w = X_features.dot(self.w)
            
            # Calculate loss (mean squared error)
            error = h_w - y.reshape(-1, 1)
            loss = (1.0/(2*N)) * np.sum(error**2)
            
            # Calculate gradient
            grad = (1.0/N) * X_features.T.dot(error)
            
            # Add regularization if needed
            if self.reg > 0:
                loss += (self.reg / 2) * np.sum(self.w[1:] ** 2)  # Don't regularize bias
                reg_grad = np.zeros_like(self.w)
                reg_grad[1:] = self.reg * self.w[1:]  # Don't regularize bias
                grad += reg_grad
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # Calculate the loss function of the polynomial regression with order m
            # ================================================================ #
            X_features = self.gen_poly_features(X)
            
            # Calculate predictions
            h_w = X_features.dot(self.w)
            
            # Calculate loss (mean squared error)
            error = h_w - y.reshape(-1, 1)
            loss = (1.0/(2*N)) * np.sum(error**2)
            
            # Calculate gradient
            grad = (1.0/N) * X_features.T.dot(error)
            
            # Add regularization if needed
            if self.reg > 0:
                loss += (self.reg / 2) * np.sum(self.w[1:] ** 2)  # Don't regularize bias
                reg_grad = np.zeros_like(self.w)
                reg_grad[1:] = self.reg * self.w[1:]  # Don't regularize bias
                grad += reg_grad
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return loss, grad
    
    def train_LR(self, X, y, eta=1e-3, batch_size=30, num_iters=1000) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.

        Inputs:
         - X         -- numpy array of shape (N,1), features
         - y         -- numpy array of shape (N,), targets
         - eta       -- float, learning rate
         - num_iters -- integer, maximum number of iterations
         
        Returns:
         - loss_history: vector containing the loss at each training iteration.
         - self.w: optimal weights 
        """
        loss_history = []
        N,d = X.shape
        for t in np.arange(num_iters):
                X_batch = None
                y_batch = None
                # ================================================================ #
                # YOUR CODE HERE:
                # Sample batch_size elements from the training data for use in gradient descent.  
                # After sampling, X_batch should have shape: (batch_size,1), y_batch should have shape: (batch_size,)
                # The indices should be randomly generated to reduce correlations in the dataset.  
                # Use np.random.choice.  It is better to user WITHOUT replacement.
                # ================================================================ #
                indices = np.random.choice(N, batch_size, replace=False)
                X_batch = X[indices]
                y_batch = y[indices]
                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #
                loss = 0.0
                grad = np.zeros_like(self.w)
                # ================================================================ #
                # YOUR CODE HERE: 
                # evaluate loss and gradient for batch data
                # save loss as loss and gradient as grad
                # update the weights self.w
                # ================================================================ #
                loss, grad = self.loss_and_grad(X_batch, y_batch)
                self.w -= eta * grad
                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #
                loss_history.append(loss)
        return loss_history, self.w
        
    def closed_form(self, X, y):
        """
        Inputs:
        - X: N x 1 array of training data.
        - y: N x 1 array of targets
        Returns:
        - self.w: optimal weights 
        """
        m = self.m
        N,d = X.shape
        loss = 0.0
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # obtain the optimal weights from the closed form solution 
            # ================================================================ #
            X_features = self.gen_poly_features(X)
            # Closed form solution: w = (X^T X)^(-1) X^T y
            XTX = X_features.T.dot(X_features)
            XTX_inv = np.linalg.inv(XTX + self.reg * np.eye(XTX.shape[0]))
            XTy = X_features.T.dot(y)
            self.w = XTX_inv.dot(XTy)
            
            # Calculate the loss with the optimal weights
            y_pred = X_features.dot(self.w)
            error = y_pred - y.reshape(-1, 1)
            loss = (1.0/(2*N)) * np.sum(error**2)
            if self.reg > 0:
                loss += (self.reg / 2) * np.sum(self.w[1:] ** 2)
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            X_features = self.gen_poly_features(X)
            # Closed form solution: w = (X^T X)^(-1) X^T y
            XTX = X_features.T.dot(X_features)
            XTX_inv = np.linalg.inv(XTX + self.reg * np.eye(XTX.shape[0]))
            XTy = X_features.T.dot(y)
            self.w = XTX_inv.dot(XTy)
            
            # Calculate the loss with the optimal weights
            y_pred = X_features.dot(self.w)
            error = y_pred - y.reshape(-1, 1)
            loss = (1.0/(2*N)) * np.sum(error**2)
            if self.reg > 0:
                loss += (self.reg / 2) * np.sum(self.w[1:] ** 2)
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return loss, self.w
    
    
    def predict(self, X):
        """
        Inputs:
        - X: N x 1 array of training data.
        Returns:
        - y_pred: Predicted targets for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros(X.shape[0])
        m = self.m
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # PREDICT THE TARGETS OF X 
            # ================================================================ #
            X_features = self.gen_poly_features(X)
            y_pred = X_features.dot(self.w).flatten()
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            X_features = self.gen_poly_features(X)
            y_pred = X_features.dot(self.w).flatten()
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return y_pred