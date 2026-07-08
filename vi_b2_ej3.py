"""
Verificacion numerica del Ejercicio 3 (Value Iteration) del tema B.2.

Estados (0..7): s = bateria + 4*sala
  Sala 0 = vigilancia, sala 1 = recarga
  Bateria: 0=baja, 1=media-baja, 2=media-alta, 3=alta

Recompensas (depende solo del estado):
  R(0) = -10 (vigilancia, bateria baja)
  R(1) = R(2) = R(3) = +5 (vigilancia, otros niveles)
  R(4..7) = 0 (recarga, cualquier nivel)

Acciones:
  a=0: ir a vigilancia
  a=1: ir a recarga

gamma = 0.9. Inicializamos Q^0 a ceros, V^0 = 0. Tres tareas:
  1) Imprimir V^1 ... V^5.
  2) Imprimir politica pi^5 y Q^5.
  3) Iterar hasta convergencia y comprobar la afirmacion del .tex:
     "la politica se estabiliza en la iteracion 9 en (1,1,0,0,1,1,0,0)".
"""

import numpy as np

# Matriz de transicion P[s, a, s'] (16 filas s,a -> 8 columnas s')
# Reordenamos a tensor (8, 2, 8).
P_raw = np.array([
    # s a |  0     1     2     3     4     5     6     7
    [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # 0,0
    [0.90, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00],  # 0,1
    [0.17, 0.83, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # 1,0
    [0.10, 0.50, 0.00, 0.00, 0.07, 0.33, 0.00, 0.00],  # 1,1
    [0.00, 0.17, 0.83, 0.00, 0.00, 0.00, 0.00, 0.00],  # 2,0
    [0.00, 0.07, 0.33, 0.00, 0.00, 0.10, 0.50, 0.00],  # 2,1
    [0.00, 0.00, 0.17, 0.83, 0.00, 0.00, 0.00, 0.00],  # 3,0
    [0.00, 0.00, 0.03, 0.17, 0.00, 0.00, 0.13, 0.67],  # 3,1
    [0.08, 0.02, 0.00, 0.00, 0.75, 0.15, 0.00, 0.00],  # 4,0
    [0.00, 0.00, 0.00, 0.00, 0.83, 0.17, 0.00, 0.00],  # 4,1
    [0.00, 0.33, 0.07, 0.00, 0.00, 0.50, 0.10, 0.00],  # 5,0
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.83, 0.17, 0.00],  # 5,1
    [0.00, 0.00, 0.50, 0.10, 0.00, 0.00, 0.33, 0.07],  # 6,0
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.83, 0.17],  # 6,1
    [0.00, 0.00, 0.00, 0.80, 0.00, 0.00, 0.00, 0.20],  # 7,0
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # 7,1
])

P = P_raw.reshape(8, 2, 8)  # [s, a, s']
R = np.array([-10, 5, 5, 5, 0, 0, 0, 0], dtype=float)  # solo depende de s
GAMMA = 0.9


def vi_step(V_prev):
    """Una iteracion de Value Iteration. Devuelve (Q_new, V_new, pi_new)."""
    Q = np.zeros((8, 2))
    for s in range(8):
        for a in range(2):
            Q[s, a] = R[s] + GAMMA * np.sum(P[s, a] * V_prev)
    V = Q.max(axis=1)
    pi = Q.argmax(axis=1)
    return Q, V, pi


def main():
    V = np.zeros(8)
    print(f"V^0 = {V}\n")
    for k in range(1, 6):
        Q, V, pi = vi_step(V)
        Vs = "  ".join(f"{v:+7.2f}" for v in V)
        print(f"V^{k} = [{Vs}]")
    print()
    print("Q^5 columnas (a=0, a=1):")
    Q5_a0 = "  ".join(f"{q:+7.2f}" for q in Q[:, 0])
    Q5_a1 = "  ".join(f"{q:+7.2f}" for q in Q[:, 1])
    print(f"  a=0: [{Q5_a0}]")
    print(f"  a=1: [{Q5_a1}]")
    print(f"\npi^5 = {tuple(int(x) for x in pi)}")

    # Iterar hasta convergencia
    print("\n--- Iterando hasta convergencia ---")
    V = np.zeros(8)
    pi_prev = None
    for k in range(1, 200):
        Q, V_new, pi = vi_step(V)
        if pi_prev is not None and np.array_equal(pi, pi_prev):
            # Convergio la politica
            pass
        V = V_new
        if k <= 15:
            print(f"k={k:2d}  pi={tuple(int(x) for x in pi)}  V={[f'{v:+.2f}' for v in V]}")
        pi_prev = pi.copy()

    # Una vez estable
    print(f"\npi final = {tuple(int(x) for x in pi)}")


if __name__ == "__main__":
    main()
