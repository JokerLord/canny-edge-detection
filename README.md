# Детектор границ Канни
___
Программа реализует следующие алгоритмы:
- Вычисление модуля градиента, используя свёртки с производными функции Гаусса.
- Подавление немаксимумов модуля градиента (составная часть алгоритма Канни).
- Конечный результат алгоритма детектирования контуров Канни.
### Вычисление модуля градиента. Запуск:
    python main.py grad (sigma) (input_file) (output_file)
### Результат немаксимального подавления. Запуск:
    python main.py nonmax (sigma) (input_file) (output_file)
### Детектирование границ с помощью алгоритма Канни. Первый параметр — сигма для вычисления градиента, следующие два параметра — вещественные числа — больший и меньший пороги соответственно. Запуск:
    python main.py canny (sigma) (thr_high) (thr_low)
