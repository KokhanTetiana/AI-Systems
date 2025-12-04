import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.5, random_state=0
)

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

ypred = regr.predict(Xtest)

coef_rounded = np.round(regr.coef_, 2)
intercept_rounded = round(regr.intercept_, 2)

print("Коефіцієнти регресії (заокруглені):", coef_rounded)
print("Інтерсепт (вільний член, заокруглений):", intercept_rounded)

print("\nПоказники якості лінійної регресії:")
print("R2 score =", round(r2_score(ytest, ypred), 2))
print("Mean absolute error (MAE) =", round(mean_absolute_error(ytest, ypred), 2))
print("Mean squared error (MSE) =", round(mean_squared_error(ytest, ypred), 2))

fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0), label='Прогнози моделі')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4, label='Ідеальний прогноз')
ax.set_xlabel('Виміряно (Справжня прогресія захворювання)')
ax.set_ylabel('Передбачено (Прогнозована прогресія захворювання)')
ax.set_title('Лінійна регресія на наборі даних про діабет')
ax.legend()
plt.show()
