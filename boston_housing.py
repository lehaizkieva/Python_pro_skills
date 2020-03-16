# Пример нейросети для регрессии
import numpy as np
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense

# Задаем seed для повторяемости результатов
numpy.random.seed(42)

# Загружаем дание
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Стандартизируем данние
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std


# Создаем нейронную сеть
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Обучаем
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)
