# Intelligent Robotics: ejercicios de RL

Verificaciones numéricas de ejercicios de aprendizaje por refuerzo de la asignatura de Robótica Inteligente (Grado en Ciberseguridad e Inteligencia Artificial, UMA).

## Scripts

- `vi_b2_ej3.py`: Value Iteration en el problema del robot con batería (tema B.2, ejercicio 3). Imprime V^1..V^5, la política y itera hasta convergencia.
- `qlearning_b3.py`: Q-learning tabular sobre una traza de experiencias (tema B.3).
- `lab8_solucion.py`: Lab 8 (planning bajo incertidumbre). Value Iteration más simulación de tres políticas (optimal, exploración, human) con histograma y boxplot de recompensas.

`vi_b2_ej3.py` y `qlearning_b3.py` solo necesitan numpy y se ejecutan sin nada más. `lab8_solucion.py` usa el módulo `taskdef` y la matriz de transición del laboratorio, que no se incluyen aquí por ser material del curso.

## Requisitos

Python 3, numpy y matplotlib (este último solo para `lab8_solucion.py`).
