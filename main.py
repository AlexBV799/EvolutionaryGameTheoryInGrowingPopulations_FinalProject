import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm


# ============================================================
# Utilidades de tiempo
# ============================================================

def format_seconds(seconds):
    """Formato legible para tiempos."""
    if seconds < 60:
        return f"{seconds:.2f} s"

    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {sec:.1f} s"

    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)} h {int(minutes)} min {sec:.1f} s"


# ============================================================
# Modelo del paper:
# Evolutionary game theory in growing populations
# Cooperadores A vs defectores B
# ============================================================

def fitness_A(x, b=3.0, c=1.0, s=0.05):
    """
    f_A(x) = 1 + s[(b-c)x - c(1-x)]
    """
    return 1.0 + s * ((b - c) * x - c * (1.0 - x))


def fitness_B(x, b=3.0, s=0.05):
    """
    f_B(x) = 1 + s*b*x
    """
    return 1.0 + s * b * x


def global_growth(x, p=10.0):
    """
    g(x) = 1 + p*x
    """
    return 1.0 + p * x


def death_rate_global(N, K=100.0):
    """
    d(N) = N/K
    """
    return N / K


# ============================================================
# Simulación estocástica Gillespie
# ============================================================

def gillespie_one_run(
    N0=4,
    x0=0.5,
    b=3.0,
    c=1.0,
    s=0.05,
    K=100.0,
    p=10.0,
    t_max=8.0,
    rng=None,
):
    """
    Simula una trayectoria estocástica con eventos:

    A -> A + 1
    B -> B + 1
    A -> A - 1
    B -> B - 1

    Las tasas son:

    nacimiento A: g(x) * f_A(x) * A
    nacimiento B: g(x) * f_B(x) * B
    muerte A:     d(N) * A
    muerte B:     d(N) * B
    """
    if rng is None:
        rng = np.random.default_rng()

    A = int(round(N0 * x0))
    B = int(N0 - A)
    t = 0.0

    times = [t]
    As = [A]
    Bs = [B]

    while t < t_max and (A + B) > 0:
        N = A + B
        x = A / N

        fA = fitness_A(x, b=b, c=c, s=s)
        fB = fitness_B(x, b=b, s=s)
        g = global_growth(x, p=p)
        d = death_rate_global(N, K=K)

        rate_birth_A = g * fA * A
        rate_birth_B = g * fB * B
        rate_death_A = d * A
        rate_death_B = d * B

        rates = np.array(
            [rate_birth_A, rate_birth_B, rate_death_A, rate_death_B],
            dtype=float,
        )

        total_rate = rates.sum()

        if total_rate <= 0:
            break

        # Tiempo hasta el próximo evento
        dt = rng.exponential(1.0 / total_rate)
        t += dt

        # Elegimos evento
        event = rng.choice(4, p=rates / total_rate)

        if event == 0:
            A += 1
        elif event == 1:
            B += 1
        elif event == 2:
            A = max(A - 1, 0)
        elif event == 3:
            B = max(B - 1, 0)

        times.append(t)
        As.append(A)
        Bs.append(B)

    return np.array(times), np.array(As), np.array(Bs)


def interpolate_trajectory(times, As, Bs, t_grid):
    """
    Interpola una trayectoria de saltos a un grid común.
    Usa valor constante hacia adelante.
    """
    idx = np.searchsorted(times, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(times) - 1)

    A_grid = As[idx]
    B_grid = Bs[idx]
    N_grid = A_grid + B_grid

    x_grid = np.divide(
        A_grid,
        N_grid,
        out=np.zeros_like(A_grid, dtype=float),
        where=N_grid > 0,
    )

    return A_grid, B_grid, N_grid, x_grid


def ensemble_simulation(
    N0=4,
    x0=0.5,
    b=3.0,
    c=1.0,
    s=0.05,
    K=100.0,
    p=10.0,
    t_max=8.0,
    n_runs=10000,
    n_points=400,
    seed=123,
    show_progress=True,
    progress_desc=None,
):
    """
    Promedio de ensamble con tqdm.

    Importante:
    El paper promedia la fracción de cooperadores como

        x_promedio = sum_i A_i / sum_i N_i

    no como promedio aritmético simple de x_i.
    """
    rng = np.random.default_rng(seed)
    t_grid = np.linspace(0, t_max, n_points)

    sum_A = np.zeros_like(t_grid)
    sum_N = np.zeros_like(t_grid)

    iterator = range(n_runs)

    if show_progress:
        if progress_desc is None:
            progress_desc = f"N0={N0}"

        iterator = tqdm(
            iterator,
            desc=progress_desc,
            unit="run",
            leave=True,
        )

    start_time = time.perf_counter()

    for _ in iterator:
        times, As, Bs = gillespie_one_run(
            N0=N0,
            x0=x0,
            b=b,
            c=c,
            s=s,
            K=K,
            p=p,
            t_max=t_max,
            rng=rng,
        )

        A_grid, B_grid, N_grid, x_grid = interpolate_trajectory(
            times,
            As,
            Bs,
            t_grid,
        )

        sum_A += A_grid
        sum_N += N_grid

    elapsed = time.perf_counter() - start_time

    mean_N = sum_N / n_runs
    mean_x = np.divide(
        sum_A,
        sum_N,
        out=np.zeros_like(sum_A),
        where=sum_N > 0,
    )

    if show_progress:
        print(f"Tiempo para N0={N0}: {format_seconds(elapsed)}")

    return t_grid, mean_N, mean_x, elapsed


# ============================================================
# Solución determinista
# ============================================================

def deterministic_solution(
    N0=4,
    x0=0.5,
    b=3.0,
    c=1.0,
    s=0.05,
    K=100.0,
    p=10.0,
    t_max=8.0,
    n_points=400,
):
    """
    Ecuaciones deterministas del paper:

    dx/dt = -s(1 + p x) x(1 - x)

    dN/dt = [(1 + p x)<f> - N/K] N
    """
    def rhs(t, y):
        x, N = y

        fA = fitness_A(x, b=b, c=c, s=s)
        fB = fitness_B(x, b=b, s=s)
        f_avg = x * fA + (1.0 - x) * fB

        dx = -s * (1.0 + p * x) * x * (1.0 - x)
        dN = ((1.0 + p * x) * f_avg - N / K) * N

        return [dx, dN]

    t_eval = np.linspace(0, t_max, n_points)

    sol = solve_ivp(
        rhs,
        [0, t_max],
        [x0, N0],
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12,
    )

    return sol.t, sol.y[1], sol.y[0]


# ============================================================
# Tiempo de cooperación
# ============================================================

def cooperation_time(t, x):
    """
    Define t_c como el tiempo durante el cual x(t) permanece
    por encima de x(0), luego de haber aumentado.

    Si nunca sube respecto de x(0), devuelve 0.
    """
    x_initial = x[0]

    if np.all(x <= x_initial + 1e-12):
        return 0.0

    above = x > x_initial
    indices = np.where(above)[0]

    if len(indices) == 0:
        return 0.0

    last = indices[-1]
    return t[last]


# ============================================================
# Figura 1 aproximada
# ============================================================

def reproduce_figure_1():
    """
    Reproduce aproximadamente la Fig. 1 del paper.

    Parámetros del paper:
    x0 = 0.5
    b = 3
    c = 1
    s = 0.05
    K = 100
    p = 10
    N0 = 2, 4, 12
    """
    total_start = time.perf_counter()

    params = dict(
        x0=0.5,
        b=3.0,
        c=1.0,
        s=0.05,
        K=100.0,
        p=10.0,
        t_max=8.0,
        n_runs=10000,
        n_points=400,
    )

    N0_values = [2, 4, 12]

    results = {}
    timings = {}

    print("Iniciando simulación de Fig. 1...")
    print(f"Corridas por cada N0: {params['n_runs']}")

    for N0 in tqdm(N0_values, desc="Valores de N0", unit="N0"):
        t, mean_N, mean_x, elapsed = ensemble_simulation(
            N0=N0,
            seed=100 + N0,
            show_progress=True,
            progress_desc=f"Simulando N0={N0}",
            **params,
        )

        results[N0] = (t, mean_N, mean_x)
        timings[N0] = elapsed

    print("Calculando solución determinista para N0=4...")

    td, Nd, xd = deterministic_solution(
        N0=4,
        x0=params["x0"],
        b=params["b"],
        c=params["c"],
        s=params["s"],
        K=params["K"],
        p=params["p"],
        t_max=params["t_max"],
        n_points=params["n_points"],
    )

    # --------------------------------------------------------
    # Fig. 1a: tamaño poblacional
    # --------------------------------------------------------
    plt.figure(figsize=(7, 4))

    for N0 in N0_values:
        t, mean_N, mean_x = results[N0]
        plt.plot(t, mean_N, label=f"Estocástico N0={N0}")

    plt.plot(td, Nd, "k--", label="Determinista N0=4")
    plt.axhline(params["K"], linestyle=":", color="black", label="K")

    plt.xlabel("tiempo t")
    plt.ylabel("tamaño poblacional N")
    plt.title("Fig. 1a aproximada: tamaño poblacional promedio")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure_1a.png", dpi=1200)
    plt.show()
    
    # --------------------------------------------------------
    # Fig. 1b: fracción de cooperadores
    # --------------------------------------------------------
    plt.figure(figsize=(7, 4))

    for N0 in N0_values:
        t, mean_N, mean_x = results[N0]
        tc = cooperation_time(t, mean_x)
        plt.plot(t, mean_x, label=f"Estocástico N0={N0}, tc≈{tc:.2f}")

        if tc > 0:
            plt.axvline(tc, linestyle=":", alpha=0.5)

    plt.plot(td, xd, "k--", label="Determinista N0=4")
    plt.axhline(params["x0"], linestyle=":", color="black", label="x0")

    plt.xlabel("tiempo t")
    plt.ylabel("fracción de cooperadores x")
    plt.title("Fig. 1b aproximada: aumento transitorio de cooperación")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure_1b.png", dpi=1200)
    plt.show()

    total_elapsed = time.perf_counter() - total_start

    print("\nResumen de tiempos Fig. 1:")
    for N0, elapsed in timings.items():
        print(f"  N0={N0}: {format_seconds(elapsed)}")

    print(f"Tiempo total Fig. 1: {format_seconds(total_elapsed)}")

    return results, timings


# ============================================================
# Figura 2 aproximada
# ============================================================

def scan_cooperation_time(
    N0_values=np.arange(2, 36, 2),
    s_values=np.linspace(0.02, 0.4, 30),
    x0=0.5,
    b=3.0,
    c=1.0,
    K=100.0,
    p=10.0,
    t_max=8.0,
    n_runs=1000,
    n_points=300,
    seed=999,
):
    """
    Escaneo para Fig. 2 con tqdm.

    Muestra:
    - barra externa para s
    - barra interna para N0
    - tiempo de cada combinación
    - tiempo total

    Para una figura más precisa, subir n_runs a 5000 o 10000.
    """
    total_start = time.perf_counter()

    tc_matrix = np.zeros((len(s_values), len(N0_values)))
    timing_matrix = np.zeros((len(s_values), len(N0_values)))

    print("Iniciando escaneo Fig. 2...")
    print(f"Total de combinaciones: {len(s_values) * len(N0_values)}")
    print(f"Corridas por combinación: {n_runs}")

    for i, s in enumerate(tqdm(s_values, desc="Escaneando s", unit="s")):
        inner_iterator = tqdm(
            N0_values,
            desc=f"N0 para s={s:.3f}",
            unit="N0",
            leave=False,
        )

        for j, N0 in enumerate(inner_iterator):
            combo_start = time.perf_counter()

            t, mean_N, mean_x, elapsed = ensemble_simulation(
                N0=N0,
                x0=x0,
                b=b,
                c=c,
                s=s,
                K=K,
                p=p,
                t_max=t_max,
                n_runs=n_runs,
                n_points=n_points,
                seed=seed + i * 1000 + j,
                show_progress=False,
            )

            tc_matrix[i, j] = cooperation_time(t, mean_x)

            combo_elapsed = time.perf_counter() - combo_start
            timing_matrix[i, j] = combo_elapsed

            tqdm.write(
                f"s={s:.3f}, N0={N0}: "
                f"tc={tc_matrix[i, j]:.3f}, "
                f"tiempo={format_seconds(combo_elapsed)}"
            )

    total_elapsed = time.perf_counter() - total_start

    print("\nResumen escaneo Fig. 2:")
    print(f"Tiempo total: {format_seconds(total_elapsed)}")
    print(f"Tiempo promedio por combinación: {format_seconds(timing_matrix.mean())}")
    print(f"Combinación más lenta: {format_seconds(timing_matrix.max())}")
    print(f"Combinación más rápida: {format_seconds(timing_matrix.min())}")

    return N0_values, s_values, tc_matrix, timing_matrix


def reproduce_figure_2():
    """
    Reproduce aproximadamente la Fig. 2 del paper.

    Ojo:
    Esto puede tardar bastante. Para probar rápido, bajá n_runs,
    achicá N0_values o achicá s_values.
    """
    N0_values, s_values, tc_matrix, timing_matrix = scan_cooperation_time(
        n_runs=1000,
    )

    plt.figure(figsize=(7, 5))

    plt.imshow(
        tc_matrix,
        origin="lower",
        aspect="auto",
        extent=[
            N0_values.min(),
            N0_values.max(),
            s_values.min(),
            s_values.max(),
        ],
    )

    plt.colorbar(label="tiempo de cooperación tc")
    plt.xlabel("tamaño inicial N0")
    plt.ylabel("fuerza de selección s")
    plt.title("Fig. 2 aproximada: tc(s, N0)")

    # Frontera asintótica del suplemento:
    # s*N0 ≈ p/(1 + p*x0)
    x0 = 0.5
    p = 10.0
    N_line = np.linspace(N0_values.min(), N0_values.max(), 200)
    s_line = p / ((1.0 + p * x0) * N_line)

    plt.plot(
        N_line,
        s_line,
        "k--",
        label=r"$sN_0 \approx p/(1+p x_0)$",
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig("figure_2.png", dpi=1200)
    plt.show()

    return N0_values, s_values, tc_matrix, timing_matrix


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    program_start = time.perf_counter()

    # --------------------------------------------------------
    # Reproduce Fig. 1
    # --------------------------------------------------------
    results_fig1, timings_fig1 = reproduce_figure_1()

    # --------------------------------------------------------
    # Reproduce Fig. 2
    # --------------------------------------------------------
    # Puede tardar bastante. Descomentá para correrla.
    #
    #N0_values, s_values, tc_matrix, timing_matrix = reproduce_figure_2()

    program_elapsed = time.perf_counter() - program_start

    print(f"\nTiempo total de ejecución del script: {format_seconds(program_elapsed)}")