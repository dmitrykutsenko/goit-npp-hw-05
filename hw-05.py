import numpy as np

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
