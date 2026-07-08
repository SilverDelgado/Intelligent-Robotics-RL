import numpy as np
P_raw = np.array([
    [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.90, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00],
    [0.17, 0.83, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.10, 0.50, 0.00, 0.00, 0.07, 0.33, 0.00, 0.00],
    [0.00, 0.17, 0.83, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.07, 0.33, 0.00, 0.00, 0.10, 0.50, 0.00],
    [0.00, 0.00, 0.17, 0.83, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.03, 0.17, 0.00, 0.00, 0.13, 0.67],
    [0.08, 0.02, 0.00, 0.00, 0.75, 0.15, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.83, 0.17, 0.00, 0.00],
    [0.00, 0.33, 0.07, 0.00, 0.00, 0.50, 0.10, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.83, 0.17, 0.00],
    [0.00, 0.00, 0.50, 0.10, 0.00, 0.00, 0.33, 0.07],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.83, 0.17],
    [0.00, 0.00, 0.00, 0.80, 0.00, 0.00, 0.00, 0.20],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
])
P = P_raw.reshape(8, 2, 8)
R = np.array([-10, 5, 5, 5, 0, 0, 0, 0], dtype=float)
GAMMA = 0.9
def vi_step(V_prev):
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
    print("\n--- Iterando hasta convergencia ---")
    V = np.zeros(8)
    pi_prev = None
    for k in range(1, 200):
        Q, V_new, pi = vi_step(V)
        if pi_prev is not None and np.array_equal(pi, pi_prev):
            pass
        V = V_new
        if k <= 15:
            print(f"k={k:2d}  pi={tuple(int(x) for x in pi)}  V={[f'{v:+.2f}' for v in V]}")
        pi_prev = pi.copy()
    print(f"\npi final = {tuple(int(x) for x in pi)}")
if __name__ == "__main__":
    main()
