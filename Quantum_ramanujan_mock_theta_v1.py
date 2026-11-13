# Quantum_ramanujan_mock_theta 



import random
import numpy as np

random.seed(39)
np.random.seed(39)

from pathlib import Path
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector



FILES = (
    Path("/data/loto7_4718_k90.csv"),
    Path("/data/loto7_4718_k90_loto_2981.csv"),
    Path("/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_BALLS = 39
DRAW_SIZE = 7
N_QUBITS = 6
N_STATES = 2 ** N_QUBITS


def load_draws(csv_path: Path) -> np.ndarray:
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
            nums = [int(x) for x in parts[:DRAW_SIZE]]
            if len(nums) != DRAW_SIZE:
                raise ValueError(f"Neispravan red u {csv_path}: {line}")
            if any(n < 1 or n > NUM_BALLS for n in nums) or len(set(nums)) != DRAW_SIZE:
                raise ValueError(f"Nelegalna kombinacija u {csv_path}: {nums}")
            rows.append(sorted(nums))
    if len(rows) < 100:
        raise ValueError(f"Premalo kola u {csv_path}: {len(rows)}")
    return np.array(rows, dtype=np.int32)


def draws_to_binary_matrix(draws: np.ndarray) -> np.ndarray:
    t_len = len(draws)
    mat = np.zeros((t_len, NUM_BALLS), dtype=np.float64)
    rows_idx = np.arange(t_len)[:, None]
    mat[rows_idx, draws - 1] = 1.0
    return mat


def compute_restricted_partition_weights() -> np.ndarray:
    max_sum = sum(range(NUM_BALLS - DRAW_SIZE + 1, NUM_BALLS + 1))
    dp = np.zeros((DRAW_SIZE + 1, max_sum + 1), dtype=np.float64)
    dp.put(0, 1.0)

    for m in range(1, NUM_BALLS + 1):
        for k in range(DRAW_SIZE, 0, -1):
            dp[k, m:] += dp[k - 1, :-m]

    total_partitions = float(np.sum(dp[DRAW_SIZE]))
    part_density_by_sum = dp[DRAW_SIZE] / (total_partitions + 1e-12)
    return part_density_by_sum


def mock_theta_f_series(q_vals: np.ndarray, n_terms: int = 12) -> np.ndarray:
    res = np.ones_like(q_vals, dtype=np.float64)
    pochhammer = np.ones_like(q_vals, dtype=np.float64)
    for n in range(1, n_terms + 1):
        pochhammer *= (1.0 + (q_vals ** n))
        term = (q_vals ** (n * n)) / ((pochhammer ** 2) + 1e-12)
        res += term
    return res


def mock_theta_shadow_series(q_vals: np.ndarray, n_terms: int = 12) -> np.ndarray:
    res = np.zeros_like(q_vals, dtype=np.float64)
    for n in range(1, n_terms + 1):
        sign = -1.0 if (n % 2 == 1) else 1.0
        res += sign * (q_vals ** (n * (n + 1) // 2))
    return res


def compute_causal_ramanujan_features(
    binary_mat: np.ndarray, q_base: float = 0.88
) -> tuple[np.ndarray, np.ndarray]:
    t_len, n_balls = binary_mat.shape
    part_density_by_sum = compute_restricted_partition_weights()
    ball_values = np.arange(1, n_balls + 1, dtype=np.float64)

    mock_f_mat = np.zeros((t_len, n_balls), dtype=np.float64)
    shadow_mat = np.zeros((t_len, 25), dtype=np.float64)

    wait_times = np.zeros(n_balls, dtype=np.float64)
    shadow_idx = np.arange(1, 26, dtype=np.float64)

    for t in range(t_len):
        curr = binary_mat[t]
        wait_times = np.where(curr > 0.5, 0.0, wait_times + 1.0)

        q_balls = q_base ** (1.0 + 0.25 * wait_times)
        f_vals = mock_theta_f_series(q_balls, n_terms=10)
        f_norm = f_vals / (float(np.sum(f_vals)) + 1e-12)
        mock_f_mat[t] = f_norm

        draw_sum = int(round(float(curr @ ball_values)))
        draw_sum = max(0, min(len(part_density_by_sum) - 1, draw_sum))
        p_sum_density = float(part_density_by_sum[draw_sum])

        q_shadow = (q_base ** (0.5 + 2.0 * p_sum_density)) ** (shadow_idx / 12.0)
        s_vals = mock_theta_shadow_series(q_shadow, n_terms=10)
        shadow_mat[t] = s_vals

    return mock_f_mat, shadow_mat


def encode_ramanujan_to_64_amplitudes(
    mock_f_t: np.ndarray,
    shadow_t: np.ndarray,
    alpha_f: float,
    beta_shadow: float,
) -> np.ndarray:
    logits_39 = np.clip(alpha_f * np.log(mock_f_t + 1e-12), -8.0, 8.0)
    exp_39 = np.exp(logits_39 - float(np.max(logits_39)))
    probs_39 = exp_39 / float(np.sum(exp_39))

    logits_25 = np.clip(beta_shadow * shadow_t, -8.0, 8.0)
    exp_25 = np.exp(logits_25 - float(np.max(logits_25)))
    probs_25 = exp_25 / float(np.sum(exp_25))

    active_share = float(NUM_BALLS) / float(N_STATES)
    shadow_share = 1.0 - active_share

    amp_64 = np.zeros(N_STATES, dtype=np.float64)
    amp_64[:NUM_BALLS] = np.sqrt(active_share * probs_39)
    amp_64[NUM_BALLS:] = np.sqrt(shadow_share * probs_25)

    amp_64 /= float(np.linalg.norm(amp_64))
    return amp_64


def quantum_modular_shadow_circuit(
    amp_64: np.ndarray, q_params: np.ndarray
) -> np.ndarray:
    qc = QuantumCircuit(N_QUBITS)
    qc.initialize(amp_64.tolist(), list(range(N_QUBITS)))

    idx = 0
    for _ in range(2):
        for q in range(N_QUBITS):
            qc.ry(float(q_params[idx]), q)
            idx += 1
            qc.rz(float(q_params[idx]), q)
            idx += 1
        for q in range(N_QUBITS - 1):
            qc.cx(q, q + 1)
        qc.cx(N_QUBITS - 1, 0)

    sv = Statevector.from_instruction(qc)
    probs_64 = np.abs(sv.data) ** 2

    dist_39 = probs_64[:NUM_BALLS].copy()
    dist_39 = np.maximum(dist_39, 1e-12)
    dist_39 /= float(np.sum(dist_39))
    return dist_39


def predict_step_distribution(
    mock_f_t: np.ndarray,
    shadow_t: np.ndarray,
    full_params: np.ndarray,
) -> np.ndarray:
    alpha_f, beta_shadow = full_params[:2]
    q_params = full_params[2:]
    amp_64 = encode_ramanujan_to_64_amplitudes(
        mock_f_t=mock_f_t,
        shadow_t=shadow_t,
        alpha_f=float(alpha_f),
        beta_shadow=float(beta_shadow),
    )
    return quantum_modular_shadow_circuit(amp_64, q_params)


def train_ramanujan_quantum_model(
    mock_f_seq: np.ndarray,
    shadow_seq: np.ndarray,
    target_binary_seq: np.ndarray,
    reg_lambda: float = 1.5e-3,
    max_iter: int = 45,
) -> np.ndarray:
    n_samples = len(mock_f_seq)
    target_dist = target_binary_seq / float(DRAW_SIZE)

    step = max(1, n_samples // 180)
    sub_f = mock_f_seq[::step]
    sub_s = shadow_seq[::step]
    sub_tgt = target_dist[::step]
    sub_len = len(sub_f)

    n_q_params = 2 * N_QUBITS * 2
    rng = np.random.RandomState(39)
    init_head = np.array([0.80, 0.25], dtype=np.float64)
    init_q = rng.uniform(-0.20, 0.20, size=n_q_params)
    init_params = np.concatenate([init_head, init_q])

    def loss_fn(params: np.ndarray) -> float:
        total_loss = 0.0
        for i in range(sub_len):
            pred_dist = predict_step_distribution(sub_f[i], sub_s[i], params)
            total_loss -= float(np.sum(sub_tgt[i] * np.log(pred_dist)))
        total_loss /= float(sub_len)
        total_loss += reg_lambda * float(np.sum(params[2:] ** 2))
        return float(total_loss)

    res = minimize(
        loss_fn,
        init_params,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-5},
    )
    return res.x


def evaluate_distribution(
    mock_f_seq: np.ndarray,
    shadow_seq: np.ndarray,
    target_binary_seq: np.ndarray,
    params: np.ndarray,
) -> tuple[float, float]:
    n_samples = len(mock_f_seq)
    target_dist = target_binary_seq / float(DRAW_SIZE)
    nll = 0.0
    hits = 0.0
    for i in range(n_samples):
        pred_dist = predict_step_distribution(mock_f_seq[i], shadow_seq[i], params)
        nll -= float(np.sum(target_dist[i] * np.log(pred_dist)))
        top7 = np.argsort(pred_dist)[-DRAW_SIZE:]
        actual = np.flatnonzero(target_binary_seq[i] > 0.5)
        hits += float(len(set(top7.tolist()).intersection(actual.tolist())))
    return nll / float(n_samples), hits / float(n_samples)


def select_next_combination(dist_39: np.ndarray) -> list[int]:
    if dist_39.shape != (NUM_BALLS,) or not np.all(np.isfinite(dist_39)):
        raise ValueError("Neispravna izlazna raspodela modela.")
    if abs(float(np.sum(dist_39)) - 1.0) > 1e-5:
        raise ValueError("Raspodela nije normalizovana.")
    top7_idx = np.argsort(dist_39)[-DRAW_SIZE:]
    combo = sorted(int(i + 1) for i in top7_idx)
    if len(combo) != DRAW_SIZE or len(set(combo)) != DRAW_SIZE:
        raise ValueError("Nelegalna izlazna kombinacija.")
    return combo


def run_pipeline_for_csv(csv_path: Path) -> None:
    draws = load_draws(csv_path)
    binary_mat = draws_to_binary_matrix(draws)
    t_len = len(binary_mat)

    n_trans = t_len - 1
    train_end = int(n_trans * 0.70)
    val_end = int(n_trans * 0.85)

    candidate_q_bases = (0.84, 0.92)

    best_q = next(iter(candidate_q_bases))
    best_val_nll = float("inf")
    best_val_hits = 0.0
    best_test_nll = float("inf")
    best_test_hits = 0.0

    for q_val in candidate_q_bases:
        f_all, s_all = compute_causal_ramanujan_features(binary_mat, q_base=q_val)

        f_train = f_all[:train_end]
        s_train = s_all[:train_end]
        y_train = binary_mat[1 : train_end + 1]

        f_val = f_all[train_end:val_end]
        s_val = s_all[train_end:val_end]
        y_val = binary_mat[train_end + 1 : val_end + 1]

        params_cv = train_ramanujan_quantum_model(
            f_train, s_train, y_train, max_iter=35
        )
        val_nll, val_hits = evaluate_distribution(f_val, s_val, y_val, params_cv)

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_val_hits = val_hits
            best_q = q_val

            f_test = f_all[val_end:n_trans]
            s_test = s_all[val_end:n_trans]
            y_test = binary_mat[val_end + 1 : n_trans + 1]
            best_test_nll, best_test_hits = evaluate_distribution(
                f_test, s_test, y_test, params_cv
            )

        f_final, s_final = compute_causal_ramanujan_features(binary_mat, q_base=best_q)

    f_full = f_final[:n_trans]
    s_full = s_final[:n_trans]
    y_full = binary_mat[1 : n_trans + 1]
    final_params = train_ramanujan_quantum_model(
        f_full, s_full, y_full, max_iter=50
    )

    next_dist = predict_step_distribution(f_final[-1], s_final[-1], final_params)
    next_combo = select_next_combination(next_dist)

    print(f"CSV: {csv_path.name} (kola: {t_len})")
    print(f"Validacija -> NLL: {best_val_nll:.4f} | Prosek pogodaka: {best_val_hits:.3f}")
    print(f"Test       -> NLL: {best_test_nll:.4f} | Prosek pogodaka: {best_test_hits:.3f}")
    print(f"NEXT       -> {next_combo}")
    print("-" * 52)


def main() -> None:
    print("Metoda: Ramanujan Mock Theta q-Series + Restricted Partitions (6 qubita)")
    print("Posebnost: Lazne teta funkcije, modularna senka i kvantna interferencija amplituda")
    print("=" * 52)
    for csv_path in FILES:
        run_pipeline_for_csv(csv_path)


if __name__ == "__main__":
    main()



"""
Metoda: Ramanujan Mock Theta q-Series + Restricted Partitions (6 qubita)
Posebnost: Lazne teta funkcije, modularna senka i kvantna interferencija amplituda
====================================================

CSV: loto7_4718_k90.csv (kola: 4718)
Validacija -> NLL: 3.6670 | Prosek pogodaka: 1.232
Test       -> NLL: 3.6663 | Prosek pogodaka: 1.280
NEXT       -> [2, x, 12, y, 31, z, 33]
----------------------------------------------------

CSV: loto7_4718_k90_loto_2981.csv (kola: 2981)
Validacija -> NLL: 3.6684 | Prosek pogodaka: 1.275
Test       -> NLL: 3.6669 | Prosek pogodaka: 1.271
NEXT       -> [2, x, 7, y, 10, z, 31]
----------------------------------------------------

CSV: loto7_4718_k90_loto_plus_1737.csv (kola: 1737)
Validacija -> NLL: 3.6714 | Prosek pogodaka: 1.177
Test       -> NLL: 3.6676 | Prosek pogodaka: 1.295
NEXT       -> [2, x, 4, y, 6, z, 12]
----------------------------------------------------
"""



"""
Srinivasa Ramanujan — Lost Notebook, lažne teta funkcije / mock theta functions, particije i q-redovi.


1. Ramanujanova izgubljena beležnica (Lost Notebook, 138 stranica): Rukopis iz 1920. godine koji je George Andrews pronašao 1976. u Kembridžu, sa preko 600 formula iz teorije brojeva, kombinatorike i analize.
2. Lažne teta funkcije (Mock Theta Functions) i q-redovi: Specijalni beskonačni q-redovi (npr. \(f(q) = \sum_{n=0}^{\infty} \frac{q^{n^2}}{(-q; q)_n^2}\)) koji se asimptotski ponašaju slično modularnim formama blizu korena jedinice, ali zahtevaju nemodularni korekcioni član ("senku" / shadow) da bi postali harmonijske Maasove forme.
3. Teorija particija i kongruencije: Brojanje načina na koje se ceo broj N može napisati kao zbir pozitivnih celih brojeva (uključujući ograničene particije na tačno k=7 delova i najviše veličine 39), zajedno sa Hardy-Ramanujan-Rademacher asimptotskim formulama.

Kombinatorna struktura ograničenih particija i q-redova prirodno opisuje unutrašnju geometriju skupa od 7 različitih brojeva iz \(\{1, \dots, 39\}\) bez kozmetičkih filtera i bez svođenja na proste frekvencije.

• Ograničene particije p(S, 7, 39) kao strukturni prostor stanja: Svako kolo Loto 7/39 sa zbirom \(S = \sum_{i=1}^7 x_i\) predstavlja jednu particiju broja S na tačno 7 različitih delova ≤ 39. Umesto da zabranjujem ili filtriramo zbirove, Ramanujanov q-razvoj \(\prod_{m=1}^{39} (1 + z q^m)\) generiše tačnu kombinatornu gustinu svih legalnih 7-članih podskupova za svako energetsko/zbirno stanje S.
• Mock Theta q-redovi kao višeskalni temporalni korelacioni filter: Evolucija razmaka između pojavljivanja svakog broja (vreme čekanja \(\tau_b(t)\) od poslednjeg izvlačenja broja b) može se preslikati preko Ramanujanovih q-Pochhammer simbola \((a; q)_n\) i lažnih teta redova f(q), φ(q), ψ(q), gde parametar q ∈ (0, 1) kontroliše nelinearno slabljenje i oscilacije faze kroz istoriju kola — bez nametanja normalne ili uniformne raspodele.
• Kvantno kodiranje q-amplituda i modularne "senke" (Qiskit, 6 qubita): U 6-qubitnom kolu (64 bazna stanja), prvih 39 stanja nosi amplitude izvedene iz Ramanujanovog q-reda i kombinatorne težine particija, dok preostalih 25 stanja kodira nemodularni korekcioni član ("senku" lažne teta funkcije) koji interferira kroz spregnute rotacije u Qiskit Statevector simulatoru, dajući direktnu regresiju raspodele za sledeće kolo.

Ramanujan Mock Theta q-redovi + kombinatorika ograničene particije u 6-qubitnom kvantnom kolu 
"""
