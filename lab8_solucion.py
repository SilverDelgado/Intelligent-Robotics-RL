import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
LABS = os.path.join(HERE, "..", "fuentes", "labs")
sys.path.insert(0, LABS)
from taskdef import State, Action, Reward, NROOMS, S, A, EpFinishedState
T_FILE = os.path.join(LABS, "T_brokenrobot.npy")
T = np.load(T_FILE)
print(f"T cargada: shape={T.shape}, suma={T.sum():.1f}")
R = np.zeros(S)
final_mask = np.zeros(S, dtype=bool)
valid_mask = np.zeros(S, dtype=bool)
for sind in range(S):
    st = State(sind)
    co, _ = st.canOccur()
    if not co:
        continue
    valid_mask[sind] = True
    if st.isFinalSuccess():
        final_mask[sind] = True
        R[sind] = 10.0
    else:
        R[sind] = -1.0
print(f"Estados validos: {valid_mask.sum()} / {S}")
print(f"Estados finales (exito): {final_mask.sum()}")
def VI(gamma, T, R, epsilon, final_mask, max_iter=5000):
    S_, A_, _ = T.shape
    V = np.zeros(S_)
    for k in range(1, max_iter + 1):
        Q = R[:, None] + gamma * np.einsum("ijk,k->ij", T, V)
        Q[final_mask] = R[final_mask, None]
        V_new = Q.max(axis=1)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < epsilon:
            print(f"VI converge en {k} iteraciones (delta={delta:.4f})")
            return Q, k
    print(f"VI no convergio tras {max_iter} iteraciones (delta={delta:.4f})")
    return Q, max_iter
GAMMA = 0.9
EPSILON = 0.3
Q, iters = VI(GAMMA, T, R, EPSILON, final_mask)
def policy_optimal(qs, sind):
    return list(np.flatnonzero(qs == qs.max()))
def policy_pure_exploration(qs, sind):
    return list(range(len(qs)))
def policy_human(qs, sind):
    st = State(sind)
    unvisited = [r for r in range(NROOMS) if not st.roomVisited(r)]
    if unvisited:
        return unvisited
    return list(range(len(qs)))
MAX_CYCLE_REPS = 5
MAX_STEPS = 200
def simulate_episode(T, Q, policy_fn, rng):
    s = State()
    s.fillRandomInitial(rng.integers)
    accrew = 0.0
    rep = [s.toIndex()]
    for step in range(MAX_STEPS):
        if s.isFinalSuccess():
            return 0, accrew, step
        sind = s.toIndex()
        qs = Q[sind, :]
        candidates = list(policy_fn(qs, sind))
        s_next = None
        while candidates:
            aind = candidates[rng.integers(0, len(candidates))]
            ts = T[sind, aind, :]
            if ts.sum() < 0.98:
                candidates = [a for a in candidates if a != aind]
                continue
            s_next_ind = rng.choice(len(ts), p=ts)
            s_next = State(int(s_next_ind))
            break
        if s_next is None:
            return 1, accrew, step
        rew = Reward()
        rew.expectedForSA(s_next, Action(int(aind)))
        accrew += rew.value()
        rep.append(s_next.toIndex())
        if len(rep) >= MAX_CYCLE_REPS:
            last = rep[-MAX_CYCLE_REPS:]
            if len(set(last)) == 1:
                return 2, accrew, step
            rep = rep[-(MAX_CYCLE_REPS - 1):]
        s = s_next
    return 2, accrew, MAX_STEPS
POLITICAS = [
    ("optimal", policy_optimal),
    ("pure-exploration", policy_pure_exploration),
    ("human", policy_human),
]
N_EPISODES = 100
SEED = 42
print("\n--- Simulacion ---")
resultados = {}
for nombre, fn in POLITICAS:
    rng = np.random.default_rng(SEED)
    accs = []
    successes = 0
    no_trans = 0
    cycles = 0
    for ep in range(N_EPISODES):
        succ, acc, nsteps = simulate_episode(T, Q, fn, rng)
        accs.append(acc)
        if succ == 0:
            successes += 1
        elif succ == 1:
            no_trans += 1
        else:
            cycles += 1
    accs = np.array(accs)
    resultados[nombre] = accs
    print(f"  {nombre:<20s}  mean={accs.mean():+7.2f}  median={np.median(accs):+7.2f}  "
          f"exito={successes}/{N_EPISODES}  ciclos={cycles}  no-trans={no_trans}")
OUT_DIR = os.path.join(HERE, "..", "aux", "figuras", "b2")
os.makedirs(OUT_DIR, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
colors = ["#1f78b4", "#a6cee3", "#33a02c"]
bins = np.linspace(min(r.min() for r in resultados.values()) - 5,
                   max(r.max() for r in resultados.values()) + 5, 20)
for ax, (nombre, accs), c in zip(axes, resultados.items(), colors):
    ax.hist(accs, bins=bins, color=c, edgecolor="black", linewidth=0.4)
    ax.set_title(f"{nombre}\nmedia = {accs.mean():+.2f}")
    ax.set_xlabel("Recompensa acumulada")
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel(f"N episodios (de {N_EPISODES})")
plt.tight_layout()
fig_hist = os.path.join(OUT_DIR, "lab8_histograma.png")
plt.savefig(fig_hist, dpi=150)
plt.close()
print(f"\nFigura histograma: {fig_hist}")
fig, ax = plt.subplots(figsize=(6, 4))
data = [resultados[n] for n, _ in POLITICAS]
bp = ax.boxplot(data, labels=[n for n, _ in POLITICAS], patch_artist=True)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
ax.set_ylabel("Recompensa acumulada por episodio")
ax.set_title(f"Comparativa de politicas (robot broken, n={N_EPISODES})")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig_box = os.path.join(OUT_DIR, "lab8_boxplot.png")
plt.savefig(fig_box, dpi=150)
plt.close()
print(f"Figura boxplot: {fig_box}")
print("\n--- Listo ---")
