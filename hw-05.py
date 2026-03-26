import numpy as np
import timeit
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


# 1. Генеруємо 100 випадкових значень ознак x1, x2

# Фіксуємо seed для відтворюваності
np.random.seed(42)

# 1.1. Генеруємо 100 випадкових точок (x1, x2)
X = np.random.uniform(-5, 5, size=(100, 2))
x1 = X[:, 0]
x2 = X[:, 1]

# 1.2. Обчислюємо y за поліномом
y = 4*x1**2 + 5*x2**2 - 2*x1*x2 + 3*x1 - 6*x2

# Перевіримо форми
print("X shape:", X.shape)
print("y shape:", y.shape)


#2. Генеруємо додаткові ознаки для кожного степеня
from sklearn.preprocessing import PolynomialFeatures

# Створюємо генератор поліноміальних ознак 2-го степеня
poly = PolynomialFeatures(degree=2, include_bias=True)

# Генеруємо нову матрицю ознак
X_poly = poly.fit_transform(X)

print("Original X shape:", X.shape)
print("Polynomial X shape:", X_poly.shape)
print("Feature names:", poly.get_feature_names_out())

# Перед п.3 потрібно спочатку провести генерацію данних для реалізації вимірів у п.4

np.random.seed(42)

X = np.random.uniform(-5, 5, size=(100, 2))
x1 = X[:, 0]
x2 = X[:, 1]

y = 4*x1**2 + 5*x2**2 - 2*x1*x2 + 3*x1 - 6*x2

poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)

print(X_poly.shape)

# 3. Базові допоміжні функції

def predict(X, w):
    return X @ w

def mse_loss(X, y, w):
    y_pred = predict(X, w)
    return np.mean((y_pred - y) ** 2)

def mse_gradient(X, y, w):
    n = X.shape[0]
    y_pred = predict(X, w)
    return (2.0 / n) * X.T @ (y_pred - y)

# 3.1. Звичайний batch gradient descent
def polynomial_regression_gradient_descent(
    X, y, 
    lr=0.01, 
    n_iters=1000
):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    losses = []

    for i in range(n_iters):
        grad = mse_gradient(X, y, w)
        w -= lr * grad
        loss = mse_loss(X, y, w)
        losses.append(loss)
    return w, losses

# 3.2. Stochastic Gradient Descent (SGD, з mini-batch)
def polynomial_regression_SGD(
    X, y, 
    lr=0.01, 
    n_iters=1000, 
    batch_size=1,
    shuffle=True
):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    losses = []

    for it in range(n_iters):
        if shuffle:
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]

            y_pred = X_batch @ w
            grad = (2.0 / X_batch.shape[0]) * X_batch.T @ (y_pred - y_batch)
            w -= lr * grad

        loss = mse_loss(X, y, w)
        losses.append(loss)
    return w, losses

# 3.3. RMSProp
def polynomial_regression_rmsprop(
    X, y, 
    lr=0.01, 
    n_iters=1000, 
    beta=0.9, 
    eps=1e-8
):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    Eg2 = np.zeros(n_features)  # накопичена середня квадр. градієнта
    losses = []

    for i in range(n_iters):
        grad = mse_gradient(X, y, w)

        Eg2 = beta * Eg2 + (1 - beta) * (grad ** 2)
        w -= lr * grad / (np.sqrt(Eg2) + eps)

        loss = mse_loss(X, y, w)
        losses.append(loss)
    return w, losses

# 3.4. Adam
def polynomial_regression_adam(
    X, y, 
    lr=0.01, 
    n_iters=1000, 
    beta1=0.9, 
    beta2=0.999, 
    eps=1e-8
):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    m = np.zeros(n_features)
    v = np.zeros(n_features)
    losses = []

    for t in range(1, n_iters + 1):
        grad = mse_gradient(X, y, w)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        w -= lr * m_hat / (np.sqrt(v_hat) + eps)

        loss = mse_loss(X, y, w)
        losses.append(loss)
    return w, losses

# 3.5. Nadam (Nesterov-accelerated Adam)
def polynomial_regression_nadam(
    X, y, 
    lr=0.01, 
    n_iters=1000, 
    beta1=0.9, 
    beta2=0.999, 
    eps=1e-8
):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    m = np.zeros(n_features)
    v = np.zeros(n_features)
    losses = []

    for t in range(1, n_iters + 1):
        grad = mse_gradient(X, y, w)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Nadam update: Nesterov-style корекція моменту
        nesterov_m = beta1 * m_hat + (1 - beta1) * grad / (1 - beta1 ** t)

        w -= lr * nesterov_m / (np.sqrt(v_hat) + eps)

        loss = mse_loss(X, y, w)
        losses.append(loss)
    return w, losses


# 4. Вимірювання часу роботи методів градієнтного спуску

# 4.1. Batch Gradient Descent
time = timeit.timeit(
    stmt="polynomial_regression_gradient_descent(X_poly, y, lr=0.001, n_iters=2000)",
    globals=globals(),
    number=5
)
print("Batch Gradient Descent average time:", time / 5)

# 4.2. Stochastic Gradient Descent (SGD)
time = timeit.timeit(
    stmt="polynomial_regression_SGD(X_poly, y, lr=0.001, n_iters=50, batch_size=1)",
    globals=globals(),
    number=5
)
print("Batch Gradient Descent average time:", time / 5)

# 4.3. RMSProp
time = timeit.timeit(
    stmt="polynomial_regression_rmsprop(X_poly, y, lr=0.001, n_iters=2000)",
    globals=globals(),
    number=5
)
print("Batch Gradient Descent average time:", time / 5)

# 4.4. Adam
time = timeit.timeit(
    stmt="polynomial_regression_adam(X_poly, y, lr=0.01, n_iters=2000)",
    globals=globals(),
    number=5
)
print("Batch Gradient Descent average time:", time / 5)

# 4.5. Nadam
time = timeit.timeit(
    stmt="polynomial_regression_nadam(X_poly, y, lr=0.01, n_iters=2000)",
    globals=globals(),
    number=5
)
print("Batch Gradient Descent average time:", time / 5)

# 5. Підбір оптимальної кількісті ітерацій для кожного з варіантів метода

def find_optimal_iterations(losses, threshold=1e-6):
    """
    Повертає номер ітерації, після якої зміни loss стають дуже малими.
    threshold — мінімальна зміна, яку вважаємо суттєвою.
    """
    for i in range(1, len(losses)):
        if abs(losses[i] - losses[i-1]) < threshold:
            return i
    return len(losses)

# 5.1. Batch Gradient Descent
w_gd, losses_gd = polynomial_regression_gradient_descent(
    X_poly, y, lr=0.001, n_iters=3000
)

opt_gd = find_optimal_iterations(losses_gd)
print("Optimal GD iterations:", opt_gd)

plt.plot(losses_gd)
plt.title("GD Loss Curve")
plt.show()

# 5.2. SGD
w_sgd, losses_sgd = polynomial_regression_SGD(
    X_poly, y, lr=0.001, n_iters=200, batch_size=1
)

opt_sgd = find_optimal_iterations(losses_sgd, threshold=1e-4)
print("Optimal SGD iterations:", opt_sgd)

plt.plot(losses_sgd)
plt.title("SGD Loss Curve")
plt.show()

# 5.3 RMSProp
w_rms, losses_rms = polynomial_regression_rmsprop(
    X_poly, y, lr=0.001, n_iters=3000
)

opt_rms = find_optimal_iterations(losses_rms)
print("Optimal RMSProp iterations:", opt_rms)

plt.plot(losses_rms)
plt.title("RMSProp Loss Curve")
plt.show()

# 5.4. Adam
w_adam, losses_adam = polynomial_regression_adam(
    X_poly, y, lr=0.01, n_iters=3000
)

opt_adam = find_optimal_iterations(losses_adam)
print("Optimal Adam iterations:", opt_adam)

plt.plot(losses_adam)
plt.title("Adam Loss Curve")
plt.show()

# 5.5. Nadam
w_nadam, losses_nadam = polynomial_regression_nadam(
    X_poly, y, lr=0.01, n_iters=3000
)

opt_nadam = find_optimal_iterations(losses_nadam)
print("Optimal Nadam iterations:", opt_nadam)

plt.plot(losses_nadam)
plt.title("Nadam Loss Curve")
plt.show()
