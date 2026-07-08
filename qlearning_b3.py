import numpy as np
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
    if sp == 0:
        return -10.0
    if sp is None:
        if a == 0:
            return -1.0
        else:
            return -10.0
    if a == 0:
        return -1.0
    return 0.0
def main():
    Q = np.zeros((3, 2))
    visitas = np.zeros((3, 2), dtype=int)
    print("Estado inicial: Q = ceros\n")
    for k, (s, a, sp) in enumerate(EXPERIENCIAS):
        alpha = 1.0 / (k + 1)
        r = recompensa(s, a, sp)
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
