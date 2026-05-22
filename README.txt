Evolutionary Game Theory in Growing Populations

Description:
This project simulates the stochastic and deterministic dynamics of cooperation in growing populations under evolutionary game theory.
The main script, `main.py`, reproduces the figures used in the final project by running Gillespie ensemble simulations and deterministic ODE integration.

Dependencies:
- numpy==2.4.4
- matplotlib==3.10.9
- scipy==1.17.1
- tqdm==4.67.3

Usage:
1. Activate the Python virtual environment for the workspace.
2. Install dependencies with:
   pip install -r requirements.txt
3. Run the main script:
   python main.py

What it does:
- `main.py` reproduces Fig. 1, Fig. 2, and Fig. 2b.
- Fig. 1: compares stochastic ensemble averages to deterministic dynamics for initial populations `N0 = 2, 4, 12`.
- Fig. 2: scans cooperation time `t_c` over a range of selection strengths `s` and initial population sizes `N0`.
- Fig. 2b: plots cooperation time curves for selected `N0` values.

Output files:
- `Figure_1a.png`, `Figure_1b.png`, `Figure_1c.png` (from Fig. 1)
- `figure_2.png` (from Fig. 2)
- `figure_2b_2.png` (from Fig. 2b)
- `figure_1_results.txt` (cached Fig. 1 results)
- `figure_2_results_N0_2_40_s60_runs1000.txt` (cached Fig. 2 results)

Notes:
- The script uses caching for figure data to avoid recomputing expensive simulations on repeat runs.
- Delete the cached files to force recomputation.
