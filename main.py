import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
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

def fitness_A(x, b=3.0, c=1.0, s=0.05): # f_A(x)
    return 1.0 + s * ((b - c) * x - c * (1.0 - x))


def fitness_B(x, b=3.0, s=0.05): # f_B(x)
    return 1.0 + s * b * x


def global_growth(x, p=10.0): #g(x)
    return 1.0 + p * x


def death_rate_global(N, K=100.0): # d(N)
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

        dt = rng.exponential(1.0 / total_rate)
        t += dt

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
# Guardado/carga de resultados Fig. 1 en .txt
# ============================================================

def save_figure_1_results_txt(
    filename,
    results,
    deterministic_results,
    timings,
    params,
    total_elapsed,
):
    """
    Guarda los resultados de Fig. 1 en un .txt.

    Columnas:
    N0, t, mean_N_stochastic, mean_x_stochastic, N_deterministic, x_deterministic, elapsed_for_N0
    """
    rows = []

    for N0 in sorted(results.keys()):
        t, mean_N, mean_x = results[N0]
        td, Nd, xd = deterministic_results[N0]
        elapsed = timings.get(N0, np.nan)

        if not np.allclose(t, td):
            raise ValueError(f"El grid temporal estocástico y determinista no coincide para N0={N0}")

        for k in range(len(t)):
            rows.append(
                [
                    N0,
                    t[k],
                    mean_N[k],
                    mean_x[k],
                    Nd[k],
                    xd[k],
                    elapsed,
                ]
            )

    rows = np.array(rows, dtype=float)

    header = (
        "Resultados Fig. 1\n"
        f"x0={params['x0']}\n"
        f"b={params['b']}\n"
        f"c={params['c']}\n"
        f"s={params['s']}\n"
        f"K={params['K']}\n"
        f"p={params['p']}\n"
        f"t_max={params['t_max']}\n"
        f"n_runs={params['n_runs']}\n"
        f"n_points={params['n_points']}\n"
        f"total_elapsed_seconds={total_elapsed}\n"
        "columns: N0 t mean_N_stochastic mean_x_stochastic N_deterministic x_deterministic elapsed_for_N0"
    )

    np.savetxt(filename, rows, header=header)
    print(f"\nResultados guardados en: {filename}")


def load_figure_1_results_txt(filename):
    """
    Carga resultados previamente guardados de Fig. 1.

    Devuelve:
    results, deterministic_results, timings
    """
    data = np.loadtxt(filename)

    results = {}
    deterministic_results = {}
    timings = {}

    N0_values = np.unique(data[:, 0]).astype(int)

    for N0 in N0_values:
        sub = data[data[:, 0] == N0]

        t = sub[:, 1]
        mean_N = sub[:, 2]
        mean_x = sub[:, 3]
        Nd = sub[:, 4]
        xd = sub[:, 5]
        elapsed = sub[0, 6]

        results[N0] = (t, mean_N, mean_x)
        deterministic_results[N0] = (t, Nd, xd)
        timings[N0] = elapsed

    print(f"\nResultados cargados desde: {filename}")

    return results, deterministic_results, timings


# ============================================================
# Figura 1 aproximada
# ============================================================

def reproduce_figure_1(
    use_cache=True,
    force_recompute=False,
    cache_filename="figure_1_results.txt",
):
    """
    Fig. 1 modificada:

    Fig. 1a:
        Subplot de 3 figuras, una para cada N0.
        En cada subplot compara N(t) estocástico vs N(t) determinista.
        Una única leyenda global.

    Fig. 1b:
        Tamaño poblacional N(t) vs tiempo para las estocásticas.

    Fig. 1c:
        Fracción de cooperadores x(t), equivalente a la anterior Fig. 1b.

    Además guarda/carga resultados en un .txt para evitar recomputar.
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

    if use_cache and (not force_recompute) and os.path.exists(cache_filename):
        results, deterministic_results, timings = load_figure_1_results_txt(cache_filename)
    else:
        results = {}
        deterministic_results = {}
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

            td, Nd, xd = deterministic_solution(
                N0=N0,
                x0=params["x0"],
                b=params["b"],
                c=params["c"],
                s=params["s"],
                K=params["K"],
                p=params["p"],
                t_max=params["t_max"],
                n_points=params["n_points"],
            )

            deterministic_results[N0] = (td, Nd, xd)

        total_elapsed_so_far = time.perf_counter() - total_start

        save_figure_1_results_txt(
            filename=cache_filename,
            results=results,
            deterministic_results=deterministic_results,
            timings=timings,
            params=params,
            total_elapsed=total_elapsed_so_far,
        )

    # --------------------------------------------------------
    # Fig. 1a:
    # Subplot de 3 figuras, una por N0.
    # Compara stochastic vs determinista para N(t).
    # Una única leyenda.
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8),
        sharex=True,
        sharey=True,
    )

    # axes_flat[0], axes_flat[1], axes_flat[2] -> subplots para N0
    # axes_flat[3] -> panel solo para la leyenda
    axes_flat = axes.ravel()

    legend_handles = []
    legend_labels = []

    for i, (ax, N0) in enumerate(zip(axes_flat[:3], N0_values)):
        t, mean_N, mean_x = results[N0]
        td, Nd, xd = deterministic_results[N0]

        line_stoch, = ax.plot(
            t,
            mean_N,
            linewidth=2,
            label="Stochastic",
        )

        line_det, = ax.plot(
            td,
            Nd,
            "k--",
            linewidth=2,
            label="Deterministic",
        )

        ax.axhline(
            params["K"],
            linestyle=":",
            color="black",
            linewidth=1,
            alpha=0.8,
        )

        ax.set_title(fr"$N_0 = {N0}$")
        ax.set_xlabel("Time t")
        ax.set_ylabel("Population size N")
        ax.grid(alpha=0.25)

        # Guardamos los handles solo una vez
        if i == 0:
            legend_handles = [line_stoch, line_det]
            legend_labels = ["Stochastic", "Deterministic"]

    # Cuarto panel: solo leyenda
    legend_ax = axes_flat[3]
    legend_ax.axis("off")

    legend_ax.legend(
        legend_handles,
        legend_labels,
        loc="center",
        frameon=True,
        fontsize=15,
    )

    fig.suptitle("Overshoot in the population size", fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("Fig_1a.png", dpi=1200, bbox_inches="tight")
    plt.show()

    # --------------------------------------------------------
    # Fig. 1b:
    # Tamaño poblacional N(t) vs tiempo para las estocásticas.
    # --------------------------------------------------------
    plt.figure(figsize=(7, 4))

    for N0 in N0_values:
        t, mean_N, mean_x = results[N0]
        plt.plot(t, mean_N, linewidth=2, label=fr"$N_0={N0}$")

    plt.axhline(params["K"], linestyle=":", color="black", label="K")

    plt.xlabel("Time t")
    plt.ylabel("Population size N")
    plt.title("Average population size over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Fig_1b.png", dpi=1200)
    plt.show()

    # --------------------------------------------------------
    # Fig. 1c:
    # Fracción de cooperadores x(t).
    # --------------------------------------------------------
    plt.figure(figsize=(7, 4))

    for N0 in N0_values:
        t, mean_N, mean_x = results[N0]
        tc = cooperation_time(t, mean_x)

        line, = plt.plot(
            t,
            mean_x,
            linewidth=2,
            label=fr"$N_0={N0}$, $t_c \approx {tc:.2f}$",
        )

        color = line.get_color() # Tomamos el color de N0

        if tc > 0:
            plt.axvline(
                tc,
                linestyle=":",
                alpha=0.7,
                color=color,
            )

    # Agrego solución determinista N0=4 como en el código anterior
    td, Nd, xd = deterministic_results[4]

    plt.plot(td, xd, "k--", linewidth=2, label=r"Deterministic $N_0=4$")
    plt.axhline(params["x0"], linestyle=":", color="black", label=r"$x_0$")

    plt.xlabel("Time t")
    plt.ylabel("Cooperation fraction x")
    plt.title("Transient increase of cooperation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Fig_1c.png", dpi=1200)
    plt.show()

    total_elapsed = time.perf_counter() - total_start

    print("\nResumen de tiempos Fig. 1:")
    for N0, elapsed in timings.items():
        print(f"  N0={N0}: {format_seconds(elapsed)}")

    return results, deterministic_results, timings

# ============================================================
# Guardado/carga de resultados Fig. 2 en .txt
# ============================================================

def save_figure_2_results_txt(
    filename,
    N0_values,
    s_values,
    tc_matrix,
    timing_matrix,
    total_elapsed,
):
    """
    Guarda los resultados de Fig. 2 en un .txt.

    Columnas:
    s, N0, tc, elapsed_for_combination
    """
    rows = []

    for i, s in enumerate(s_values):
        for j, N0 in enumerate(N0_values):
            rows.append(
                [
                    s,
                    N0,
                    tc_matrix[i, j],
                    timing_matrix[i, j],
                ]
            )

    rows = np.array(rows, dtype=float)

    header = (
        "Resultados Fig. 2\n"
        f"total_elapsed_seconds={total_elapsed}\n"
        "columns: s N0 tc elapsed_for_combination"
    )

    np.savetxt(filename, rows, header=header)
    print(f"\nResultados Fig. 2 guardados en: {filename}")


def load_figure_2_results_txt(filename):
    """
    Carga resultados previamente guardados de Fig. 2.

    Devuelve:
    N0_values, s_values, tc_matrix, timing_matrix
    """
    data = np.loadtxt(filename)

    s_values = np.unique(data[:, 0])
    N0_values = np.unique(data[:, 1]).astype(int)

    tc_matrix = np.zeros((len(s_values), len(N0_values)))
    timing_matrix = np.zeros((len(s_values), len(N0_values)))

    for row in data:
        s, N0, tc, elapsed = row

        i = np.where(np.isclose(s_values, s))[0][0]
        j = np.where(N0_values == int(N0))[0][0]

        tc_matrix[i, j] = tc
        timing_matrix[i, j] = elapsed

    print(f"\nResultados Fig. 2 cargados desde: {filename}")

    return N0_values, s_values, tc_matrix, timing_matrix
# ============================================================
# Simulación para Fig. 2
# ============================================================

def scan_cooperation_time(
    N0_values=np.arange(2, 40, 2),
    s_values=np.linspace(0.02, 0.4, 60),
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

    Requiere que ya tengas definidas estas funciones en el script:
    - ensemble_simulation
    - cooperation_time
    - format_seconds
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



# ============================================================
# Limpieza física de tc_matrix
# ============================================================

def clean_tc_curve_physical(tc_values, zero_threshold=0.05):
    """
    Limpia una curva t_c(s) para un N0 fijo.

    Impone:
    1. Valores t_c <= zero_threshold se consideran cero.
    2. t_c no puede aumentar al aumentar s.
    3. Cuando t_c llega a cero, permanece cero para s mayores.

    Esto elimina picos espurios debidos a ruido estadístico.
    """
    tc_clean = np.array(tc_values, copy=True, dtype=float)

    tc_clean[tc_clean <= zero_threshold] = 0.0

    for i in range(1, len(tc_clean)):
        if tc_clean[i] > tc_clean[i - 1]:
            tc_clean[i] = tc_clean[i - 1]

    zero_indices = np.where(tc_clean <= zero_threshold)[0]
    if len(zero_indices) > 0:
        first_zero = zero_indices[0]
        tc_clean[first_zero:] = 0.0

    return tc_clean


def clean_tc_matrix_physical(tc_matrix, zero_threshold=0.05):
    """
    Aplica clean_tc_curve_physical a cada columna de tc_matrix.

    Filas: valores de s.
    Columnas: valores de N0.
    """
    tc_clean = np.array(tc_matrix, copy=True, dtype=float)

    for j in range(tc_clean.shape[1]):
        tc_clean[:, j] = clean_tc_curve_physical(
            tc_clean[:, j],
            zero_threshold=zero_threshold,
        )

    return tc_clean


# Alias por compatibilidad con código anterior.
def enforce_zero_after_first_zero(tc_values, zero_threshold=0.05):
    return clean_tc_curve_physical(tc_values, zero_threshold=zero_threshold)


def clean_tc_matrix_after_zero(tc_matrix, zero_threshold=0.05):
    return clean_tc_matrix_physical(tc_matrix, zero_threshold=zero_threshold)


# ============================================================
# Colormap azul
# ============================================================

def paper_like_blues(vmax=8.0):
    """
    Colormap azul con escala lineal.
    """
    colors = [
        (0.00 / vmax, "#f2f7fb"),
        (0.25 / vmax, "#d6e6f2"),
        (1.00 / vmax, "#8bbbd9"),
        (3.00 / vmax, "#3f8fc1"),
        (8.00 / vmax, "#084a91"),
    ]

    return mcolors.LinearSegmentedColormap.from_list(
        "paper_like_blues",
        colors,
    )


# ============================================================
# Frontera empírica y ajuste físico
# ============================================================

def estimate_empirical_boundary_from_tc(
    N0_values,
    s_values,
    tc_matrix_clean,
    tc_threshold=0.05,
):
    """
    Estima la frontera empírica desde una matriz ya limpiada.

    Para cada N0 busca la transición:
        t_c > tc_threshold  ->  t_c <= tc_threshold
    """
    boundary_N0 = []
    boundary_s = []

    for j, N0 in enumerate(N0_values):
        tc_col = tc_matrix_clean[:, j]
        active = tc_col > tc_threshold

        if not np.any(active):
            continue

        if np.all(active):
            continue

        last_active = np.where(active)[0].max()

        if last_active >= len(s_values) - 1:
            continue

        s1 = s_values[last_active]
        s2 = s_values[last_active + 1]
        tc1 = tc_col[last_active]
        tc2 = tc_col[last_active + 1]

        if np.isclose(tc1, tc2):
            s_boundary = 0.5 * (s1 + s2)
        else:
            s_boundary = s1 + (tc_threshold - tc1) * (s2 - s1) / (tc2 - tc1)

        boundary_N0.append(N0)
        boundary_s.append(s_boundary)

    return np.array(boundary_N0), np.array(boundary_s)


def fit_inverse_boundary(boundary_N0, boundary_s, fit_offset=False):
    """
    Ajusta la frontera empírica.

    Por defecto:
        s = A / N0

    Si fit_offset=True:
        s = A / (N0 + B)
    """
    boundary_N0 = np.asarray(boundary_N0, dtype=float)
    boundary_s = np.asarray(boundary_s, dtype=float)

    if len(boundary_N0) < 2:
        raise ValueError("No hay suficientes puntos para ajustar la frontera.")

    if not fit_offset:
        def model(N0, A):
            return A / N0

        popt, _ = curve_fit(
            model,
            boundary_N0,
            boundary_s,
            p0=[1.0],
            bounds=(0.0, np.inf),
            maxfev=10000,
        )
        return popt[0], 0.0

    def model_offset(N0, A, B):
        return A / (N0 + B)

    popt, _ = curve_fit(
        model_offset,
        boundary_N0,
        boundary_s,
        p0=[1.0, 0.0],
        bounds=([0.0, -10.0], [np.inf, 50.0]),
        maxfev=10000,
    )
    return popt[0], popt[1]


def inverse_boundary_curve(N0_values, A, B=0.0):
    return A / (N0_values + B)


def empirical_sc_for_N0(
    N0,
    N0_values,
    s_values,
    tc_matrix_clean,
    tc_threshold=0.05,
):
    """
    Valor crítico s_c para un N0 concreto, calculado directamente
    desde su curva t_c(s). Esto se usa para Fig. 2b.
    """
    if N0 not in N0_values:
        return None

    j = np.where(N0_values == N0)[0][0]
    tc_col = tc_matrix_clean[:, j]
    active = tc_col > tc_threshold

    if not np.any(active):
        return None

    if np.all(active):
        return None

    last_active = np.where(active)[0].max()

    if last_active >= len(s_values) - 1:
        return None

    s1 = s_values[last_active]
    s2 = s_values[last_active + 1]
    tc1 = tc_col[last_active]
    tc2 = tc_col[last_active + 1]

    if np.isclose(tc1, tc2):
        return 0.5 * (s1 + s2)

    return s1 + (tc_threshold - tc1) * (s2 - s1) / (tc2 - tc1)


# ============================================================
# Preparación común para Fig. 2 y Fig. 2b
# ============================================================

def get_fig2_prepared_data(
    cache_filename,
    use_cache=True,
    force_recompute=False,
    clean_for_plot=True,
    zero_threshold=0.05,
    tc_threshold=0.05,
):
    """
    Carga/calcula Fig. 2 y devuelve tanto la matriz cruda como la limpia.

    Fig. 2 y Fig. 2b deben usar esta función para ser consistentes.
    """
    total_start = time.perf_counter()

    if use_cache and (not force_recompute) and os.path.exists(cache_filename):
        N0_values, s_values, tc_matrix_raw, timing_matrix = load_figure_2_results_txt(
            cache_filename
        )
    else:
        N0_values, s_values, tc_matrix_raw, timing_matrix = scan_cooperation_time(
            N0_values=np.arange(2, 40, 2),
            s_values=np.linspace(0.02, 0.4, 60),
            n_runs=1000,
        )

        total_elapsed_so_far = time.perf_counter() - total_start
        save_figure_2_results_txt(
            filename=cache_filename,
            N0_values=N0_values,
            s_values=s_values,
            tc_matrix=tc_matrix_raw,
            timing_matrix=timing_matrix,
            total_elapsed=total_elapsed_so_far,
        )

    if clean_for_plot:
        tc_matrix_plot = clean_tc_matrix_physical(
            tc_matrix_raw,
            zero_threshold=zero_threshold,
        )
    else:
        tc_matrix_plot = np.array(tc_matrix_raw, copy=True)

    boundary_N0, boundary_s = estimate_empirical_boundary_from_tc(
        N0_values=N0_values,
        s_values=s_values,
        tc_matrix_clean=tc_matrix_plot,
        tc_threshold=tc_threshold,
    )

    if len(boundary_N0) >= 2:
        A_fit, B_fit = fit_inverse_boundary(
            boundary_N0,
            boundary_s,
            fit_offset=False,
        )
    else:
        A_fit, B_fit = np.nan, 0.0

    return {
        "N0_values": N0_values,
        "s_values": s_values,
        "tc_matrix_raw": tc_matrix_raw,
        "tc_matrix_plot": tc_matrix_plot,
        "timing_matrix": timing_matrix,
        "boundary_N0": boundary_N0,
        "boundary_s": boundary_s,
        "A_fit": A_fit,
        "B_fit": B_fit,
    }


# ============================================================
# Fig. 2
# ============================================================

def reproduce_figure_2(
    use_cache=True,
    force_recompute=False,
    cache_filename="figure_2_results.txt",
    clean_for_plot=True,
    zero_threshold=0.05,
    tc_threshold=0.05,
    save_filename="figure_2.png",
    show_boundary_points=False,
):
    """
    Fig. 2 físicamente consistente.

    - Mapa de color: t_c limpiado con criterio físico mínimo.
    - Puntos negros opcionales: frontera empírica extraída de esa misma matriz.
    - Línea sólida: ajuste empírico s = A/N0.
    - Línea discontinua: predicción teórica s = A_theory/N0,
      con A_theory = p/(1 + p x0) para p=10, x0=0.5.
    """
    total_start = time.perf_counter()

    data = get_fig2_prepared_data(
        cache_filename=cache_filename,
        use_cache=use_cache,
        force_recompute=force_recompute,
        clean_for_plot=clean_for_plot,
        zero_threshold=zero_threshold,
        tc_threshold=tc_threshold,
    )

    N0_values = data["N0_values"]
    s_values = data["s_values"]
    tc_matrix_plot = data["tc_matrix_plot"]
    timing_matrix = data["timing_matrix"]
    boundary_N0 = data["boundary_N0"]
    boundary_s = data["boundary_s"]
    A_fit = data["A_fit"]
    B_fit = data["B_fit"]

    fig, ax = plt.subplots(figsize=(7, 5))

    vmax_tc = 8.0
    paper_cmap = paper_like_blues(vmax=vmax_tc)

    dN = N0_values[1] - N0_values[0]
    ds = s_values[1] - s_values[0]

    extent = [
        N0_values.min() - dN / 2,
        N0_values.max() + dN / 2,
        s_values.min() - ds / 2,
        s_values.max() + ds / 2,
    ]

    im = ax.imshow(
        tc_matrix_plot,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap=paper_cmap,
        vmin=0.0,
        vmax=vmax_tc,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Cooperation time $t_c$")

    ax.set_xlabel(r"Initial population size $N_0$")
    ax.set_ylabel(r"Selection strength $s$")
    ax.set_title(r"Dependence of $t_c$ on $s$ and $N_0$")

    if show_boundary_points and len(boundary_N0) > 0:
        ax.scatter(
            boundary_N0,
            boundary_s,
            color="black",
            s=22,
            alpha=0.75,
            zorder=5,
            label="_nolegend_",
        )

    N_line = np.linspace(N0_values.min(), N0_values.max(), 1000)
    handles = []

    if np.isfinite(A_fit):
        s_fit = inverse_boundary_curve(N_line, A_fit, B_fit)
        mask_fit = (s_fit >= s_values.min()) & (s_fit <= s_values.max())

        solid_line, = ax.plot(
            N_line[mask_fit],
            s_fit[mask_fit],
            "k-",
            linewidth=2.2,
            label=fr"Empirical boundary, $sN_0 \approx {A_fit:.2f}$",
        )
        handles.append(solid_line)

    p_theory = 10.0
    x0_theory = 0.5
    A_theory = p_theory / (1.0 + p_theory * x0_theory)
    s_theory = A_theory / N_line
    mask_theory = (s_theory >= s_values.min()) & (s_theory <= s_values.max())

    dashed_line, = ax.plot(
        N_line[mask_theory],
        s_theory[mask_theory],
        "k--",
        linewidth=2.0,
        label=fr"Theory, $sN_0 \approx {A_theory:.2f}$",
    )
    handles.append(dashed_line)

    ax.set_xlim(N0_values.min() - dN / 2, N0_values.max() + dN / 2)
    ax.set_ylim(s_values.min(), s_values.max())
    ax.set_xticks([5, 10, 15, 20, 25, 30, 35])
    ax.set_yticks([0.02, 0.1, 0.2, 0.3, 0.4])

    ax.legend(handles=handles, loc="upper right")

    fig.tight_layout()
    fig.savefig(save_filename, dpi=1200, bbox_inches="tight")
    plt.show()

    total_elapsed = time.perf_counter() - total_start
    print(f"Tiempo total Fig. 2: {format_seconds(total_elapsed)}")

    return (
        N0_values,
        s_values,
        tc_matrix_plot,
        timing_matrix,
        boundary_N0,
        boundary_s,
        A_fit,
    )


# ============================================================
# Fig. 2b
# ============================================================

def reproduce_figure_2b_from_txt(
    cache_filename="figure_2_results.txt",
    selected_N0_values=(4, 6, 8),
    save_filename="figure_2b.png",
    clean_for_plot=True,
    zero_threshold=0.05,
    tc_threshold=0.05,
    show_transition_lines=True,
):
    """
    Fig. 2b: t_c frente a s para varios N0.

    Las líneas verticales marcan s_c(N0) calculado directamente desde
    cada curva t_c(s), no desde el ajuste global.
    """
    data = get_fig2_prepared_data(
        cache_filename=cache_filename,
        use_cache=True,
        force_recompute=False,
        clean_for_plot=clean_for_plot,
        zero_threshold=zero_threshold,
        tc_threshold=tc_threshold,
    )

    N0_values = data["N0_values"]
    s_values = data["s_values"]
    tc_matrix_plot = data["tc_matrix_plot"]
    timing_matrix = data["timing_matrix"]

    fig, ax = plt.subplots(figsize=(7, 4))

    for N0 in selected_N0_values:
        if N0 not in N0_values:
            print(f"Aviso: N0={N0} no está en {cache_filename}. Se ignora.")
            continue

        j = np.where(N0_values == N0)[0][0]

        line, = ax.plot(
            s_values,
            tc_matrix_plot[:, j],
            marker="o",
            linewidth=2,
            markersize=4,
            label=fr"$N_0={N0}$",
        )

        if show_transition_lines:
            s_c = empirical_sc_for_N0(
                N0=N0,
                N0_values=N0_values,
                s_values=s_values,
                tc_matrix_clean=tc_matrix_plot,
                tc_threshold=tc_threshold,
            )

            if s_c is not None and s_values.min() <= s_c <= s_values.max():
                ax.axvline(
                    s_c,
                    linestyle=":",
                    linewidth=1.3,
                    alpha=0.75,
                    color=line.get_color(),
                )

    ax.set_xlabel(r"Selection strength $s$")
    ax.set_ylabel(r"Cooperation time $t_c$")
    ax.set_title(r"$t_c$ vs $s$ for different $N_0$")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_filename, dpi=1200, bbox_inches="tight")
    plt.show()

    return N0_values, s_values, tc_matrix_plot, timing_matrix


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    program_start = time.perf_counter()

    # --------------------------------------------------------
    # Reproduce Fig. 1
    # --------------------------------------------------------

    """
    results_fig1, deterministic_fig1, timings_fig1 = reproduce_figure_1(
        use_cache=True,
        force_recompute=False,
        cache_filename="figure_1_results.txt",
    )
    """

    # --------------------------------------------------------
    # Reproduce Fig. 2 y Fig. 2b
    # --------------------------------------------------------

    filename_2 = "figure_2_results_N0_2_40_s60_runs1000.txt"

    N0_values, s_values, tc_matrix, timing_matrix, boundary_N0, boundary_s, A_fit = reproduce_figure_2(
        use_cache=True,
        force_recompute=False,
        cache_filename=filename_2,
        clean_for_plot=True,
        zero_threshold=0.05,
        tc_threshold=0.05,
        save_filename="figure_2.png",
        show_boundary_points=False,
    )

    N0_values, s_values, tc_matrix, timing_matrix = reproduce_figure_2b_from_txt(
        cache_filename=filename_2,
        selected_N0_values=(4, 6, 8),
        save_filename="figure_2b_2.png",
        clean_for_plot=True,
        zero_threshold=0.05,
        tc_threshold=0.05,
        show_transition_lines=True,
    )

    program_elapsed = time.perf_counter() - program_start
    print(f"\nTiempo total de ejecución del script: {format_seconds(program_elapsed)}")