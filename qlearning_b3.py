"""
Verificacion numerica del ejercicio de Q-learning del tema B.3 (slide 69).

Robot con bateria: 3 estados (0=Discharged terminal, 1=Medium, 2=Full),
2 acciones (0=usar Random Janitor con r=-1, 1=continuar con r=0 o r=-10 si s'=0).
gamma = 0.9, alpha_k = 1/(k+1), epsilon = 0.3.

Reglas de recompensa derivadas del enunciado:
  - Llegar al estado 0 (Discharged): recompensa -10 y fin de episodio
  - Accion 0 (Janitor): recompensa -1
  - Accion 1 a estado != 0: recompensa 0
  - Si s' = None: episodio terminado; V(s') = 0
"""

import numpy as np

# Experiencias en orden: (s, a, s')
EXPERIENCIAS = [
    (1, 1, 1),
    (1, 1, 0),
    (0, 0, None),
    (0, 1, None),
    (1, 1, None),
    (1, 0, 2),
    (2, 1, 2),
    (2, 0, 2),
    (2, 0, 2),
    (2, 0, 2),
]

GAMMA = 0.9


def recompensa(s, a, sp):
    # Llegar al estado 0 (Discharged): -10, sea con la accion que sea
    if sp == 0:
        return -10.0
    # Si sp es None: episodio termina. Si la accion fue 0 (Janitor), penalizacion -1.
    # Si la accion fue 1, asumimos r=-10 si veniamos hacia abajo (no es el caso aqui)
    # En la practica: las experiencias (s,1,None) con s != 0 se interpretan
    # como que la bateria se descargo (s' iba a ser 0) y por eso termina: r = -10
    # Y (s,0,None) con s != 0 indica un final por otro motivo: r = -1
    if sp is None:
        if a == 0:
            return -1.0
        else:
            # accion 1 que termina episodio: significa que sp era 0 (descarga)
            return -10.0
    # Sp != None y sp != 0: experiencia normal
    if a == 0:
        return -1.0  # uso del Janitor
    return 0.0  # continuar sin incidente


def main():
    Q = np.zeros((3, 2))
    visitas = np.zeros((3, 2), dtype=int)

    print("Estado inicial: Q = ceros\n")
    for k, (s, a, sp) in enumerate(EXPERIENCIAS):
        alpha = 1.0 / (k + 1)
        r = recompensa(s, a, sp)
        # V(s') = max_a Q(s', a) si s' valido y no terminal; 0 si terminal o None
        if sp is None or sp == 0:
            V_sp = 0.0
        else:
            V_sp = float(np.max(Q[sp]))
        td_target = r + GAMMA * V_sp
        Q_old = Q[s, a]
        Q[s, a] = (1 - alpha) * Q_old + alpha * td_target
        visitas[s, a] += 1
        print(f"k={k}  exp=({s},{a},{sp})  alpha={alpha:.4f}  r={r:+.2f}  V(sp)={V_sp:+.4f}")
        print(f"     Q({s},{a}): {Q_old:+.4f} -> {Q[s,a]:+.4f}")
        print(f"     Q =\n{Q}\n")

    print("=== Resultado final ===")
    print("Q =")
    print(Q)
    print("\nVisitas =")
    print(visitas)
    print("\nPolitica greedy por estado:")
    for s in range(3):
        a_best = int(np.argmax(Q[s]))
        print(f"  s={s}  argmax_a Q = {a_best}  (Q(s,0)={Q[s,0]:+.4f}, Q(s,1)={Q[s,1]:+.4f})")


if __name__ == "__main__":
    main()
