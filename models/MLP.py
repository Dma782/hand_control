import numpy as np
from dataset import X_train, y_train, X_val, y_val

class MLP:
    def __init__(
            self,
            input_size = 63,
            hidden1_size = 128,
            hidden2_size = 64,
            output_size = 3,
            seed = 1,
    ):
        rng = np.random.default_rng(seed)

        self.W1 =  rng.standard_normal((input_size, hidden1_size))* np.sqrt(2.0 / input_size) # dise 1: 63->128
        self.W2 =  rng.standard_normal((hidden1_size, hidden2_size))* np.sqrt(2.0 / hidden1_size)# dise 2: 128->64
        self.W3 =  rng.standard_normal((hidden2_size, output_size))* np.sqrt(2.0 / hidden2_size)# dise 3: 64->3

        self.b1 = np.zeros((1, hidden1_size))
        self.b2 = np.zeros((1, hidden2_size))
        self.b3 = np.zeros((1, output_size))
        self.learning_rate = 0.001

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def softmax(z):
        z = z - np.max(z, axis =1, keepdims=True)
        exp_z = np.exp(z)
        res = exp_z/np.sum(exp_z, axis=1, keepdims=True)

        return res

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)

        z3 = a2 @ self.W3 + self.b3
        y_pred = self.softmax(z3)

        cache = self.W2, self.W3, z1, z2, a1, a2

        return y_pred, cache

    def cross_entropy(self, y, y_pred):
        correct_probs = np.clip(y_pred[np.arange(len(y)), y], 1e-15, 1.0)
        loss = -np.log(correct_probs)

        return loss

    def mean_cross_entropy(self, y, y_pred):
        correct_probs = np.clip(y_pred[np.arange(len(y)), y], 1e-15, 1.0)
        loss = -np.mean(np.log(correct_probs))

        return loss

    def backward(self, X, y, y_pred, cache):
        W2, W3, z1, z2, a1, a2 = cache

        delta3 = y_pred.copy()
        delta3[np.arange(len(y)), y] -= 1
        delta3 /= len(y)

        dW3 = a2.T @ delta3
        db3 = np.sum(delta3, axis = 0, keepdims=True)

        delta2 = (delta3 @ W3.T) * (z2 > 0)
        dW2 = a1.T @ delta2
        db2 = np.sum(delta2, axis = 0, keepdims=True)

        delta1 = (delta2 @ W2.T) * (z1 > 0)
        dW1 = X.T @ delta1
        db1 = np.sum(delta1, axis = 0, keepdims=True)

        return dW1, dW2, dW3, db1, db2, db3

    def train(self, X, y, X_val, y_val, epoch_counts):
        for epoch in range(epoch_counts):
            batch_size = 32

            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

            losses = []

            for start in range(0, len(X), batch_size):
                X_batch = X[start:start + batch_size]
                y_batch = y[start:start + batch_size]

                y_pred, cache = self.forward(X_batch)
                loss = self.mean_cross_entropy(y_batch, y_pred)
                losses.append(loss)

                dW1, dW2, dW3, db1, db2, db3 = self.backward(X_batch, y_batch, y_pred, cache)

                self.W1 = self.W1 - self.learning_rate*dW1
                self.W2 = self.W2 - self.learning_rate*dW2
                self.W3 = self.W3 - self.learning_rate*dW3

                self.b1 = self.b1 - self.learning_rate*db1
                self.b2 = self.b2 - self.learning_rate*db2
                self.b3 = self.b3 - self.learning_rate*db3


            if epoch % 100 == 0:
                train_loss = np.mean(losses)
                val_loss, val_accuracy = self.evaluate(X_val, y_val)

                print(
                    f"Epoch {epoch} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Accuracy: {val_accuracy:.4f}"
                )

    def evaluate(self, X, y):
        y_pred, _ = self.forward(X)

        loss = self.mean_cross_entropy(y, y_pred)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == y)

        return loss, accuracy


if __name__ == "__main__":
    model = MLP()
    model.train(X_train, y_train, X_val, y_val, epoch_counts=500)

    np.savez(
        "model_weights.npz",
        W1=model.W1, W2=model.W2, W3=model.W3,
        b1=model.b1, b2=model.b2, b3=model.b3
    )