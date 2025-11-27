import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# -------------------------------------------------------
# КРОК 3. РОЗДІЛЕННЯ НАВЧАЛЬНОГО ТА ТЕСТОВОГО НАБОРУ
# -------------------------------------------------------
array = dataset.values
X = array[:,0:4]  # ознаки
Y = array[:,4]    # мітки класів

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1, shuffle=True)

# -------------------------------------------------------
# КРОК 4. КЛАСИФІКАЦІЯ І ПОРІВНЯННЯ АЛГОРИТМІВ
# -------------------------------------------------------
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

# -------------------------------------------------------
# КРОК 6. ПЕРЕДБАЧЕННЯ НА КОНТРОЛЬНІЙ ВИБІРЦІ
# -------------------------------------------------------
best_model = SVC(gamma='auto')
best_model.fit(X_train, Y_train)
predictions = best_model.predict(X_validation)

# -------------------------------------------------------
# КРОК 7. ОЦІНКА ЯКОСТІ МОДЕЛІ
# -------------------------------------------------------
print("Accuracy на тестовому наборі:", accuracy_score(Y_validation, predictions))
print("Confusion Matrix:")
print(confusion_matrix(Y_validation, predictions))
print("Classification Report:")
print(classification_report(Y_validation, predictions))

# -------------------------------------------------------
# КРОК 8. ПЕРЕДБАЧЕННЯ ДЛЯ НОВОЇ КВІТКИ
# -------------------------------------------------------
X_new = np.array([[5, 2.9, 1, 0.2]])
print("Форма масиву X_new:", X_new.shape)

prediction_new = best_model.predict(X_new)
print("Прогнозоване значення класу:", prediction_new)
