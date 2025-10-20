import numpy as np

class SVMRegressor:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epsilon=0.1, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param  # regularization strength
        self.epsilon = epsilon             # epsilon-insensitive margin
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                y_pred = np.dot(x_i, self.w) + self.b
                error = y_pred - y[idx]

                # If error is outside epsilon-tube
                if abs(error) > self.epsilon:
                    sign = 1 if error > 0 else -1
                    dw = 2 * self.lambda_param * self.w + sign * x_i
                    db = sign
                else:
                    # Inside epsilon zone — no loss gradient
                    dw = 2 * self.lambda_param * self.w
                    db = 0

                # Gradient descent update
                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b


# ----------------------------
# Example Run
# ----------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_regression

    # Generate synthetic regression data
    X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)

    # Train our SVM Regressor
    svr = SVMRegressor(learning_rate=0.001, lambda_param=0.01, epsilon=10, n_iters=1000)
    svr.fit(X, y)

    # Predictions
    preds = svr.predict(X)

    # Plot
    plt.scatter(X, y, color="blue", label="Data")
    plt.plot(X, preds, color="red", label="SVR fit")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("SVM Regressor (ε = 10)")
    plt.legend()
    plt.show()
