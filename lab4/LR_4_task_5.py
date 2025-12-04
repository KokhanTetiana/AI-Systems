import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(5)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.4 * X**2 + X + 4 + np.random.randn(m, 1)

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)  # X та X^2

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)

print("Коефіцієнти лінійної регресії:")
print("Intercept:", round(lin_reg.intercept_[0], 2))
print("Coef:", round(lin_reg.coef_[0][0], 2))

print("\nКоефіцієнти поліноміальної регресії:")
print("Intercept:", round(poly_reg.intercept_[0], 2))
print("Coef (X, X^2):", np.round(poly_reg.coef_[0], 2))

plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label='Випадкові дані')
plt.plot(X, y_lin_pred, color='red', linewidth=2, label='Лінійна регресія')
plt.plot(X, y_poly_pred, color='green', linewidth=2, label='Поліноміальна регресія (2-й степінь)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Лінійна та поліноміальна регресія (варіант 5)')
plt.legend()
plt.show()

mse_poly = mean_squared_error(y, y_poly_pred)
r2_poly = r2_score(y, y_poly_pred)
print("\nПоліноміальна регресія (степінь 2) оцінка:")
print("MSE =", round(mse_poly, 2))
print("R2 =", round(r2_poly, 2))
