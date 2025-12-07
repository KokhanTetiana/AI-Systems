import datetime
import json
import numpy as np
import yfinance as yf
from sklearn import covariance, cluster
from sklearn.preprocessing import RobustScaler

input_file = 'company_symbol_mapping.json'
with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())

symbols, _ = np.array(list(company_symbols_map.items())).T

start_date = datetime.datetime(2003, 7, 3)
end_date = datetime.datetime(2007, 5, 4)

print(f"Завантаження даних для {len(symbols)} компаній...")
data = yf.download(list(symbols), start=start_date, end=end_date)
print("Завантаження завершене.")

opening_quotes = data['Open']
closing_quotes = data['Close']
quotes_diff = opening_quotes - closing_quotes

print(f"\nПочаткова кількість компаній: {quotes_diff.shape[1]}")

quotes_diff.dropna(axis='columns', how='all', inplace=True)
print(f"Компаній після видалення незавантажених: {quotes_diff.shape[1]}")

quotes_diff.dropna(axis='rows', how='any', inplace=True)
print(f"Кількість днів з повними даними: {quotes_diff.shape[0]}")

remaining_symbols = quotes_diff.columns.tolist()
names = np.array([company_symbols_map[s] for s in remaining_symbols])

X = quotes_diff.copy()
scaler = RobustScaler()
X = scaler.fit_transform(X)

edge_model = covariance.GraphicalLassoCV(assume_centered=True)
edge_model.fit(X)

median_val = np.median(edge_model.covariance_)
af_model = cluster.AffinityPropagation(preference=median_val, random_state=42)
af_model.fit(edge_model.covariance_)

labels = af_model.labels_
num_labels = labels.max()
print("\n--- Результати кластеризації компаній ---")
for i in range(num_labels + 1):
    cluster_members = names[labels == i]
    print(f"Кластер {i+1} ==> {', '.join(cluster_members)}")
