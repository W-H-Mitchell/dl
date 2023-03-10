import numpy as np

class LogisticRegressionGD:
    """
    Gradient descent LR classifier
    
    Params:
    eta: float
        Set learning rate
    n_inter: int
        Passes over the training dataset
    random_state: int
        Random number generated for random weight at initiation

    Attributes:
    w_ : 1d_array of weights 
        weights following training
    b_ : scalar 
        bias unit after fitting
    losses_ : list 
        mean squared losses for each epoch
    """

    def __init__(self, eta, n_inter, random_state):
        self.eta = eta 
        self.n_inter = n_inter
        self.random_state = random_state
    

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        for i in range(n_inter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            # matrix - vector multiplication between feature matrix and error vector 
            # (transpose X matrix to match errors column vector) 
            self.w_ = self.eta * 2.0 * X.T.dot(errors) / X.shape[0]  
            self.b_ = self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output)) - 
                    ((1 - y).dot(np.log( 1- output))))
            self.losses_.append(loss)

        return self
    

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_ # dot product of x and w plus bias 
    

    def activation(self, z):
        """
        Sigmoid function activation
        """
        return 1./(1. + np.exp(-np.clip(-z,-250,250)))
    
    
    def predict(self,X):
        """
        Return class label after each unit step
        """
        return np.where(self.activation(self.net_input(X)) > 0.5, 1, 0)