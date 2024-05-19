import numpy as np

class LogisticRegression():
    def __init__(self, lr=0.001, n_iters=150):
        self.weights = []
        self.bias = 0
        self.lr = lr
        self.n_iters = n_iters
        self.losses = []
    
    def fit(self, train_data, train_target):
        self._initialize_weights(train_data.shape[1])
        for _ in range(self.n_iters):
            z = self._calculate_z(train_data)
            y_hat = self._likelihood(z)
            loss = self._compute_entropy(train_target, y_hat)
            self.losses.append(loss)
            dz = y_hat- train_target
            dw = (np.dot(train_data.transpose(), dz))* (1/train_data.shape[0])
            db = np.sum(dz) * (1/train_data.shape[0])
            self.bias = self.bias - (self.lr*db)
            self.weights = self.weights - (self.lr*dw)

    def _initialize_weights(self, weights_dim):
        self.weights = np.zeros((weights_dim))

    def _calculate_z(self, x):
        return np.dot(x,self.weights) + self.bias
    
    def _likelihood(self,z):
        return 1/(1+np.exp(-z))
    
    def _compute_entropy(self,y, y_hat):
        entropy_1 = y * np.log(y_hat)
        entropy_2 = (1-y) * np.log(1-y_hat)
        return -np.mean(entropy_1 + entropy_2)
    
    def predict(self, x):
        threshold = 0.5
        z = self._calculate_z(x)
        y_pred = self._likelihood(z)
        y_pred_cls = [1 if i > threshold else 0 for i in y_pred]
        return np.array(y_pred_cls)




