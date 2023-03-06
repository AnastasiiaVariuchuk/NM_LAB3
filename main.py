import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

# Завантаження даних
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Вибір даних для 3 класів (індекси 1, 4, 8)???????
classes = [1, 0, 2]
idx_train = np.isin(y_train_full, classes)
idx_test = np.isin(y_test, classes)
X_train, y_train = X_train_full[idx_train], y_train_full[idx_train]
X_test, y_test = X_test[idx_test], y_test[idx_test]

# Масштабування даних
X_train = X_train / 255.0
X_test = X_test / 255.0

# Створення моделі
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(len(classes), activation="softmax")
])

# Компіляція моделі з різними значеннями learning_rate
learning_rates = [0.0001, 0.001, 0.01, 0.1]
# learning_rates = [0.1]
for learning_rate in learning_rates:
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Навчання моделі
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

    # Виведення результатів для кожного learning_rate
    print(f"Learning Rate: {learning_rate}")
    print(f"Train accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

    # Побудова confusion matrix для тестового набору даних
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred.ravel().round()[:3000], normalize='all', labels=classes)
    print(f"Confusion Matrix 1:\n{cm}\n")
    cm = confusion_matrix(y_test, y_pred.ravel().round()[3000:6000:], normalize='all', labels=classes)
    print(f"Confusion Matrix 2:\n{cm}\n")
    cm = confusion_matrix(y_test, y_pred.ravel().round()[6000:], normalize='all', labels=classes)
    print(f"Confusion Matrix 3:\n{cm}\n")

if __name__ == '__main__':
    # print(history.history.keys())
    # Виведення графіку точності та втрат в процесі навчання
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


