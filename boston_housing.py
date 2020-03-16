# Пример нейросети для регрессии
import numpy as np
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense

# Задаем seed для повторяемости результатов
numpy.random.seed(42)

# Загружаем дание
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
