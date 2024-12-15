from ensembles import RandomForestMSE
from sklearn.datasets import make_regression
from scipy.sparse import csr_matrix

X, y = make_regression(
    n_samples=1000,        # Количество образцов
    n_features=10,         # Количество признаков
    noise=10.0,            # Уровень шума
    random_state=42        # Для воспроизводимости
)

# Преобразуем X в разреженную матрицу
y = y - y.min() + 1
X_sparse = csr_matrix(X)

model = RandomForestMSE(n_estimators=100, random_state=42, tree_params={'max_depth': 10})

data = model.fit(X, y, trace=True)
print(data)

