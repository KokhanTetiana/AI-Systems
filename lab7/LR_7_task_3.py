import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

X = np.loadtxt('data_clustering.txt', delimiter=',')

bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

cluster_centers = meanshift_model.cluster_centers_
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))

print(f"Координати центрів кластерів:\n{cluster_centers}")
print(f"Оцінена кількість кластерів = {num_clusters}")

plt.figure()
markers = 'o*xvsD'
marker_cycle = cycle(markers)

for i in range(num_clusters):
    cluster_points = X[labels == i]
    marker = next(marker_cycle)
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        marker=marker,
        color='black',
        s=50,
        label=f'Кластер {i}'
    )

plt.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    marker='P',
    color='red',
    s=200,
    zorder=10,
    label='Центри кластерів'
)

plt.title('Кластеризація методом зсуву середнього (Mean Shift)')
plt.xlabel('Ознака 1')
plt.ylabel('Ознака 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
