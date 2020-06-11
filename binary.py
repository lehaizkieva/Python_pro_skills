# Функция бинарного поиска элемента
# Только в отсортированом списке
import math
import random


def loop_zaloop():
	random_list = []
	for i in range(0, 100):
		x = random.randint(1, 228)		
		random_list.append(x)
	print(random_list)
	random_list.sort()
	print(random_list)
	return random_list


def binary_serch(list, item):
	low = 0
	high = len(list) - 1
	while low <= high:		
		mid = (low + high)
		guess = list[mid]
		if guess == item:
			return mid
		if guess > item:
			high = mid -1
		else:
			low = mid + 1
	return None


print (binary_serch(loop_zaloop(), 3))