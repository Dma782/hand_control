import csv
import numpy as np

# Загружаем датасет
with open("data/dataset.csv", "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# Признаки и правильные ответы
feature_names = [
    name for name in rows[0].keys()
    if name != "label"
]

X = np.array(
    [[float(row[name]) for name in feature_names] for row in rows],
    dtype=np.float64
)

labels = np.array([row["label"] for row in rows])

# Переводим названия классов в числа
class_names = ["fist", "open_palm", "peace"]

label_to_id = {
    name: i for i, name in enumerate(class_names)
}

y = np.array(
    [label_to_id[label] for label in labels],
    dtype=np.int64
)
if __name__ == "__main__":
    print("Размер X:", X.shape)
    print("Размер y:", y.shape)
    print("Классы:", np.unique(y, return_counts=True))

def train_val_test_split(X, y, seed=42):
    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []
    test_indices = []

    for class_id in np.unique(y):
        indices = np.where(y == class_id)[0]
        rng.shuffle(indices)

        n = len(indices)

        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_indices.extend(indices[:train_end])
        val_indices.extend(indices[train_end:val_end])
        test_indices.extend(indices[val_end:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    rng.shuffle(train_indices)

    return (
        X[train_indices], y[train_indices],
        X[val_indices], y[val_indices],
        X[test_indices], y[test_indices]
    )


X_train, y_train, X_val, y_val, X_test, y_test = (
    train_val_test_split(X, y)
)
if __name__ == "__main__":
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)