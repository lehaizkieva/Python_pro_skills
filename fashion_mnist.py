# Пример нейросети для классификации
from tensorflow.keras.datasets import fashion_mnist  # Модуль для работи с данними fashion_mnist
from tensorflow.keras.models import Sequential # Полседовательная можель нейронной сети - модуль
from tensorflow.keras.layers import Dense # Модуль полносвязной нейронной сети
from tensorflow.keras import utils
import numpy

# Загружаем данние x_train - изображения для обучения y_train - парвильние ответи
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Преобразования для входа в нейросеть размерности изображений
x_train = x_train.reshape(60000, 784)
# Нормализация даних
x_train /= 255

# Преобразование метки в категории
y_train = utils.to_categorical(y_train, 10)

# Список с названиями классов
classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

# Создаем последовательную модель
model = Sequential()

# Добавляем уровни сети
model.add(Dense(800, input_dim=784, activation='relu')) # Добавляем входной слой
model.add(Dense(10, activation="softmax")) # Виходной слой

# Компилируем модель с параметрами
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

print(model.summary())

# Обучаем сеть
model.fit(x_train, y_train, batch_size=200, epochs=100, verbose=1)

# Оцениваем качество обучения на тестових данних
scores = model.evaluate(x_test, y_test, verbose=1)

print("Доля верних ответов на тестових данних, в процентах:", round(scores[1] * 100, 4))

# Запускаем скть на входних данних
predictions = model.predict(x_train)

# Виводим один из результатов распознавания
print(predictions[0])

# Виводим номер класса, предсказаний нейросетью
print(np.argmax(predictions[0]))

# Виводим правильний номер класса
print(np.argmax(y_train[0]))