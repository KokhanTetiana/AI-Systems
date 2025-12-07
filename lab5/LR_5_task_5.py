import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            data.append(items)
    return np.array(data)


def encode_features(data):
    label_encoders = []
    X_encoded = np.empty(data.shape, dtype=int)

    for i in range(data.shape[1]):
        if np.all([item.isdigit() for item in data[:, i]]):
            X_encoded[:, i] = data[:, i].astype(int)
        else:
            encoder = preprocessing.LabelEncoder()
            X_encoded[:, i] = encoder.fit_transform(data[:, i])
            label_encoders.append(encoder)

    return X_encoded, label_encoders


if __name__ == '__main__':
    input_file = 'traffic_data.txt'
    data = load_data(input_file)

    X_encoded, label_encoders = encode_features(data)

    X = X_encoded[:, :-1]
    y = X_encoded[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5
    )

    regressor = ExtraTreesRegressor(n_estimators=100, max_depth=4, random_state=0)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    print(f"Mean absolute error: {mean_absolute_error(y_test, y_pred):.2f}")

    test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
    test_datapoint_encoded = []

    print("\nOriginal test data point:", test_datapoint)

    count = 0
    for i, item in enumerate(test_datapoint):
        if item.isdigit():
            test_datapoint_encoded.append(int(item))
        else:
            encoded_value = label_encoders[count].transform([item])[0]
            test_datapoint_encoded.append(int(encoded_value))
            count += 1

    print("Encoded test data point:", test_datapoint_encoded)

    predicted_traffic = regressor.predict([test_datapoint_encoded])
    print(f"\nPredicted traffic for the test data point: {int(predicted_traffic[0])}")
