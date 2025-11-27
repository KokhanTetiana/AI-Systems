from sklearn.datasets import load_iris
iris_dataset = load_iris()


print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))


print("\nОпис набору даних (DESCR):")
print(iris_dataset['DESCR'][:193] + "\n...")


print("\nНазви відповідей (сорти ірисів):")
print(iris_dataset['target_names'])


print("\nНазва ознак: ")
print(iris_dataset['feature_names'])


print("\nТип масиву data:", type(iris_dataset['data']))
print("Форма масиву data:", iris_dataset['data'].shape)


print("\nПерші 5 прикладів даних (X):")
print(iris_dataset['data'][:5])

print("\nТип масиву target:", type(iris_dataset['target']))
print("Відповіді (мітки) для всіх квіток:")
print(iris_dataset['target'][:5], "...")
print("Відповідність міток назвам сортів:")
for i, name in enumerate(iris_dataset['target_names']):
    print(f"{i} - {name}")
