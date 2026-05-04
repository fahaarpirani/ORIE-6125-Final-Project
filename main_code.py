
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

Array = np.ndarray


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def _as_float_array(x: Array) -> Array:
    return np.asarray(x, dtype=float)


def _sigmoid(z: Array) -> Array:
    z = _as_float_array(z)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _log1pexp(z: Array) -> Array:
    return np.logaddexp(0.0, z)


def _project_ball(x: Array, R: float) -> Array:
    norm_x = np.linalg.norm(x)
    if norm_x <= R or norm_x == 0.0:
        return x.copy()
    return (R / norm_x) * x


def _validate_loss(loss: str, allowed: Sequence[str]) -> None:
    if loss not in allowed:
        raise ValueError(f"loss must be one of {allowed}, got {loss!r}")


def condition_number_ata(A: Array) -> float:
    vals = np.linalg.eigvalsh(A.T @ A)
    return float(vals[-1] / vals[0])


# ---------------------------------------------------------------------
# 1. Data generation
# ---------------------------------------------------------------------

def generate_data(n: int, m: int, sigma: float, seed: int = 6365) -> Tuple[Array, Array]:
    """
    Generate A in R^{m x n} and b in R^m with cond(A^T A) approximately sigma.

    Construction:
        A = U diag(s) V^T
    where U has orthonormal columns, V is orthogonal, and singular values
    are chosen geometrically so that s_max^2 / s_min^2 = sigma.
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")
    if m < n:
        raise ValueError("This generator assumes m >= n.")
    if sigma < 1:
        raise ValueError("sigma must be at least 1.")

    rng = np.random.default_rng(seed)

    Gu = rng.normal(size=(m, n))
    U, _ = np.linalg.qr(Gu, mode="reduced")

    Gv = rng.normal(size=(n, n))
    V, _ = np.linalg.qr(Gv)

    if n == 1:
        s = np.array([1.0])
    else:
        # choose singular values so that cond(A^T A) = sigma exactly in exact arithmetic
        s_max = math.sqrt(float(sigma))
        s_min = 1.0
        exponents = np.linspace(0.0, 1.0, n)
        s = s_max * (s_min / s_max) ** exponents

    A = U @ np.diag(s) @ V.T
    b = rng.normal(size=m)
    return A, b


# ---------------------------------------------------------------------
# Smooth losses and oracles
# ---------------------------------------------------------------------

def _smooth_value_from_residual(r: Array, loss: str, mu: float, x: Array) -> float:
    if loss == "quadratic":
        loss_term = 0.5 * np.mean(r ** 2)
    elif loss == "logistic":
        loss_term = np.mean(_log1pexp(r))
    else:
        raise ValueError("smooth loss must be 'quadratic' or 'logistic'")
    reg = 0.5 * mu * float(np.dot(x, x))
    return float(loss_term + reg)


def _smooth_gradient_from_residual(A: Array, r: Array, loss: str, mu: float, x: Array) -> Tuple[Array, int]:
    m = A.shape[0]
    if loss == "quadratic":
        s = r
    elif loss == "logistic":
        s = _sigmoid(r)
    else:
        raise ValueError("smooth loss must be 'quadratic' or 'logistic'")
    grad = (A.T @ s) / m + mu * x
    return grad, 1


def _smooth_value(A: Array, b: Array, loss: str, mu: float, x: Array) -> Tuple[float, Array, int]:
    r = A @ x - b
    f = _smooth_value_from_residual(r, loss, mu, x)
    return f, r, 1


def _smooth_oracle(A: Array, b: Array, loss: str, mu: float, x: Array) -> Tuple[float, Array, Array, int]:
    f, r, count = _smooth_value(A, b, loss, mu, x)
    g, count_g = _smooth_gradient_from_residual(A, r, loss, mu, x)
    return f, g, r, count + count_g


def theory_L(A: Array, loss: str, mu: float) -> float:
    eigmax = float(np.linalg.eigvalsh(A.T @ A)[-1])
    m = A.shape[0]
    if loss == "quadratic":
        return eigmax / m + mu
    if loss == "logistic":
        return eigmax / (4.0 * m) + mu
    raise ValueError("loss must be 'quadratic' or 'logistic'")


# ---------------------------------------------------------------------
# Nonsmooth losses and oracles
# ---------------------------------------------------------------------

def _nonsmooth_value_from_residual(r: Array, loss: str) -> float:
    if loss == "quadratic":
        return float(0.5 * np.mean(r ** 2))
    if loss == "logistic":
        return float(np.mean(_log1pexp(r)))
    if loss == "l1":
        return float(np.mean(np.abs(r)))
    raise ValueError("loss must be 'quadratic', 'logistic', or 'l1'")


def _subgradient_from_residual(A: Array, r: Array, loss: str) -> Tuple[Array, int]:
    m = A.shape[0]
    if loss == "quadratic":
        s = r
    elif loss == "logistic":
        s = _sigmoid(r)
    elif loss == "l1":
        s = np.sign(r)
    else:
        raise ValueError("loss must be 'quadratic', 'logistic', or 'l1'")
    g = (A.T @ s) / m
    return g, 1


def _subgradient_oracle(A: Array, b: Array, loss: str, x: Array) -> Tuple[float, Array, Array, int]:
    r = A @ x - b
    f = _nonsmooth_value_from_residual(r, loss)
    g, count_g = _subgradient_from_residual(A, r, loss)
    return f, g, r, 1 + count_g


# ---------------------------------------------------------------------
# Histories
# ---------------------------------------------------------------------

def _empty_history() -> Dict[str, List[float]]:
    return {"func": [], "grad": [], "time": [], "mat_vec": []}


def _append_history(history: Dict[str, List[float]], f: float, gnorm: float, elapsed: float, matvec_count: int) -> None:
    history["func"].append(float(f))
    history["grad"].append(float(gnorm))
    history["time"].append(float(elapsed))
    history["mat_vec"].append(int(matvec_count))


# ---------------------------------------------------------------------
# 2.1 Gradient method
# ---------------------------------------------------------------------

def gradient_method(
    A: Array,
    b: Array,
    loss: str,
    mu: float,
    x_0: Array,
    n_iters: int,
    L_0: float,
    adaptive: bool = False,
):
    _validate_loss(loss, ("quadratic", "logistic"))
    if mu < 0:
        raise ValueError("mu must be nonnegative")
    if n_iters < 0:
        raise ValueError("n_iters must be nonnegative")
    if L_0 <= 0:
        raise ValueError("L_0 must be positive")

    x = _as_float_array(x_0).copy()
    history = _empty_history()
    status = {
        "method": "gradient",
        "adaptive": bool(adaptive),
        "converged": False,
        "iterations": 0,
        "final_L": float(L_0),
        "message": "maximum number of iterations reached",
    }

    t0 = time.perf_counter()
    mat_vec = 0

    f, g, _, count = _smooth_oracle(A, b, loss, mu, x)
    mat_vec += count
    gnorm = float(np.linalg.norm(g))
    _append_history(history, f, gnorm, time.perf_counter() - t0, mat_vec)

    best_x = x.copy()
    best_f = f
    Lk = float(L_0)

    for k in range(n_iters):
        if gnorm <= 1e-12:
            status["converged"] = True
            status["iterations"] = k
            status["message"] = "gradient norm below tolerance"
            status["final_L"] = float(Lk)
            break

        if adaptive:
            Ltrial = max(Lk / 2.0, 1e-16)
            while True:
                x_trial = x - g / Ltrial
                f_trial, r_trial, c1 = _smooth_value(A, b, loss, mu, x_trial)
                mat_vec += c1
                d = x_trial - x
                upper = f + float(np.dot(g, d)) + 0.5 * Ltrial * float(np.dot(d, d))
                if f_trial <= upper + 1e-12:
                    g_trial, c2 = _smooth_gradient_from_residual(A, r_trial, loss, mu, x_trial)
                    mat_vec += c2
                    Lk = float(Ltrial)
                    break
                Ltrial *= 2.0
        else:
            x_trial = x - g / Lk
            f_trial, r_trial, c1 = _smooth_value(A, b, loss, mu, x_trial)
            g_trial, c2 = _smooth_gradient_from_residual(A, r_trial, loss, mu, x_trial)
            mat_vec += c1 + c2

        x = x_trial
        f = float(f_trial)
        g = g_trial
        gnorm = float(np.linalg.norm(g))

        if f < best_f:
            best_f = f
            best_x = x.copy()

        _append_history(history, f, gnorm, time.perf_counter() - t0, mat_vec)
        status["iterations"] = k + 1
        status["final_L"] = float(Lk)

    return best_x, history, status


# ---------------------------------------------------------------------
# 2.2 Fast gradient method
# ---------------------------------------------------------------------

def fast_gradient_method(
    A: Array,
    b: Array,
    loss: str,
    mu: float,
    x_0: Array,
    n_iters: int,
    L_0: float,
    adaptive: bool = False,
):
    _validate_loss(loss, ("quadratic", "logistic"))
    if mu < 0:
        raise ValueError("mu must be nonnegative")
    if n_iters < 0:
        raise ValueError("n_iters must be nonnegative")
    if L_0 <= 0:
        raise ValueError("L_0 must be positive")

    x = _as_float_array(x_0).copy()
    x_prev = x.copy()
    y = x.copy()

    history = _empty_history()
    status = {
        "method": "fast_gradient",
        "adaptive": bool(adaptive),
        "converged": False,
        "iterations": 0,
        "final_L": float(L_0),
        "message": "maximum number of iterations reached",
    }

    t0 = time.perf_counter()
    mat_vec = 0

    f_x, g_x, _, count = _smooth_oracle(A, b, loss, mu, x)
    mat_vec += count
    gnorm = float(np.linalg.norm(g_x))
    _append_history(history, f_x, gnorm, time.perf_counter() - t0, mat_vec)

    best_x = x.copy()
    best_f = f_x
    Lk = float(L_0)
    tk = 1.0

    for k in range(n_iters):
        if gnorm <= 1e-12:
            status["converged"] = True
            status["iterations"] = k
            status["message"] = "gradient norm below tolerance"
            status["final_L"] = float(Lk)
            break

        if mu > 0.0 and k > 0:
            beta = (math.sqrt(Lk) - math.sqrt(mu)) / (math.sqrt(Lk) + math.sqrt(mu))
            beta = min(max(beta, 0.0), 0.999999)
            y = x + beta * (x - x_prev)
        elif mu == 0.0 and k > 0:
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
            beta = (tk - 1.0) / t_next
            y = x + beta * (x - x_prev)
        else:
            y = x.copy()

        f_y, g_y, _, c0 = _smooth_oracle(A, b, loss, mu, y)
        mat_vec += c0

        if adaptive:
            Ltrial = max(Lk / 2.0, 1e-16)
            while True:
                x_trial = y - g_y / Ltrial
                f_trial, r_trial, c1 = _smooth_value(A, b, loss, mu, x_trial)
                mat_vec += c1
                d = x_trial - y
                upper = f_y + float(np.dot(g_y, d)) + 0.5 * Ltrial * float(np.dot(d, d))
                if f_trial <= upper + 1e-12:
                    g_trial, c2 = _smooth_gradient_from_residual(A, r_trial, loss, mu, x_trial)
                    mat_vec += c2
                    Lk = float(Ltrial)
                    break
                Ltrial *= 2.0
        else:
            x_trial = y - g_y / Lk
            f_trial, r_trial, c1 = _smooth_value(A, b, loss, mu, x_trial)
            g_trial, c2 = _smooth_gradient_from_residual(A, r_trial, loss, mu, x_trial)
            mat_vec += c1 + c2

        x_old = x.copy()
        x = x_trial
        x_prev = x_old
        f_x = float(f_trial)
        g_x = g_trial
        gnorm = float(np.linalg.norm(g_x))

        if mu == 0.0:
            tk = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))

        if f_x < best_f:
            best_f = f_x
            best_x = x.copy()

        _append_history(history, f_x, gnorm, time.perf_counter() - t0, mat_vec)
        status["iterations"] = k + 1
        status["final_L"] = float(Lk)

    return best_x, history, status


# ---------------------------------------------------------------------
# 3. Projected subgradient method
# ---------------------------------------------------------------------

def subgradient_method(
    A: Array,
    b: Array,
    loss: str,
    R: float,
    x_0: Array,
    n_iters: int,
    gamma: float,
    normalized: bool = False,
):
    _validate_loss(loss, ("quadratic", "logistic", "l1"))
    if R <= 0:
        raise ValueError("R must be positive")
    if n_iters < 0:
        raise ValueError("n_iters must be nonnegative")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    x = _project_ball(_as_float_array(x_0), R)
    history = _empty_history()
    status = {
        "method": "subgradient",
        "normalized": bool(normalized),
        "converged": False,
        "iterations": 0,
        "message": "maximum number of iterations reached",
    }

    t0 = time.perf_counter()
    mat_vec = 0

    f, g, _, count = _subgradient_oracle(A, b, loss, x)
    mat_vec += count
    gnorm = float(np.linalg.norm(g))
    _append_history(history, f, gnorm, time.perf_counter() - t0, mat_vec)

    best_x = x.copy()
    best_f = f

    for k in range(n_iters):
        if gnorm <= 1e-12:
            status["converged"] = True
            status["iterations"] = k
            status["message"] = "subgradient norm below tolerance"
            break

        direction = g / gnorm if normalized and gnorm > 0.0 else g
        x = _project_ball(x - gamma * direction, R)
        f, g, _, count = _subgradient_oracle(A, b, loss, x)
        mat_vec += count
        gnorm = float(np.linalg.norm(g))

        if f < best_f:
            best_f = f
            best_x = x.copy()

        _append_history(history, f, gnorm, time.perf_counter() - t0, mat_vec)
        status["iterations"] = k + 1

    return best_x, history, status


# ---------------------------------------------------------------------
# Experiment and plotting helpers
# ---------------------------------------------------------------------

def estimate_reference_value(
    A: Array,
    b: Array,
    loss: str,
    mu: float,
    candidate_points: Sequence[Array],
    x_init: Optional[Array] = None,
    use_scipy: bool = True,
) -> float:
    vals = []
    for x in candidate_points:
        f, _, _ = _smooth_value(A, b, loss, mu, x)
        vals.append(float(f))

    if use_scipy and SCIPY_AVAILABLE:
        x0 = np.zeros(A.shape[1]) if x_init is None else _as_float_array(x_init).copy()

        def fun(x: Array) -> float:
            f, _, _ = _smooth_value(A, b, loss, mu, x)
            return f

        def jac(x: Array) -> Array:
            _, g, _, _ = _smooth_oracle(A, b, loss, mu, x)
            return g

        res = minimize(fun=fun, x0=x0, jac=jac, method="L-BFGS-B")
        if res.success:
            vals.append(float(res.fun))

    return float(min(vals))


def _residual_curve(history: Dict[str, List[float]], f_star: float) -> np.ndarray:
    y = np.asarray(history["func"], dtype=float) - float(f_star)
    return np.maximum(y, 1e-16)


def plot_histories(
    histories: Dict[str, Dict[str, List[float]]],
    f_star: Optional[float],
    title_prefix: str,
    out_dir: str,
    y_label_norm: str = "gradient / subgradient norm",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for name, hist in histories.items():
        x = np.arange(len(hist["func"]))
        y = _residual_curve(hist, f_star) if f_star is not None else np.asarray(hist["func"], dtype=float)
        if f_star is not None:
            plt.semilogy(x, y, label=name)
        else:
            plt.plot(x, y, label=name)
    plt.xlabel("iteration k")
    plt.ylabel(r"$f(x_k)-f^\star$" if f_star is not None else r"$f(x_k)$")
    plt.title(title_prefix.replace("_", " "))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{title_prefix}_residual_vs_iteration.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, hist in histories.items():
        plt.semilogy(np.arange(len(hist["grad"])), np.maximum(np.asarray(hist["grad"], dtype=float), 1e-16), label=name)
    plt.xlabel("iteration k")
    plt.ylabel(y_label_norm)
    plt.title(title_prefix.replace("_", " "))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{title_prefix}_norm_vs_iteration.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, hist in histories.items():
        y = _residual_curve(hist, f_star) if f_star is not None else np.asarray(hist["func"], dtype=float)
        if f_star is not None:
            plt.semilogy(hist["time"], y, label=name)
        else:
            plt.plot(hist["time"], y, label=name)
    plt.xlabel("time (seconds)")
    plt.ylabel(r"$f(x_k)-f^\star$" if f_star is not None else r"$f(x_k)$")
    plt.title(title_prefix.replace("_", " "))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{title_prefix}_residual_vs_time.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, hist in histories.items():
        y = _residual_curve(hist, f_star) if f_star is not None else np.asarray(hist["func"], dtype=float)
        if f_star is not None:
            plt.semilogy(hist["mat_vec"], y, label=name)
        else:
            plt.plot(hist["mat_vec"], y, label=name)
    plt.xlabel("cumulative matrix-vector products")
    plt.ylabel(r"$f(x_k)-f^\star$" if f_star is not None else r"$f(x_k)$")
    plt.title(title_prefix.replace("_", " "))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{title_prefix}_residual_vs_matvec.png", dpi=180)
    plt.close()


def run_smooth_experiments(
    n: int = 100,
    m: int = 1000,
    sigma_values: Sequence[float] = (10.0, 1e3, 1e5),
    mu_values: Sequence[float] = (0.0, 1e-3, 1e-1),
    n_iters: int = 250,
    seed: int = 6365,
    out_dir: str = "smooth_results",
    use_scipy_reference: bool = True,
) -> Dict[str, Dict[str, float]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, float]] = {}

    for sigma in sigma_values:
        A, b = generate_data(n=n, m=m, sigma=float(sigma), seed=seed)
        x0 = np.zeros(n)

        for loss in ("quadratic", "logistic"):
            for mu in mu_values:
                L0 = theory_L(A, loss, mu)
                sols = []
                hists = {}

                x_gd, hist_gd, stat_gd = gradient_method(A, b, loss, mu, x0, n_iters, L0, adaptive=False)
                x_fg, hist_fg, stat_fg = fast_gradient_method(A, b, loss, mu, x0, n_iters, L0, adaptive=False)
                x_gda, hist_gda, stat_gda = gradient_method(A, b, loss, mu, x0, n_iters, L0, adaptive=True)
                x_fga, hist_fga, stat_fga = fast_gradient_method(A, b, loss, mu, x0, n_iters, L0, adaptive=True)

                sols.extend([x_gd, x_fg, x_gda, x_fga])
                hists = {
                    "GD": hist_gd,
                    "FGM": hist_fg,
                    "GD-adaptive": hist_gda,
                    "FGM-adaptive": hist_fga,
                }

                f_star = estimate_reference_value(A, b, loss, mu, candidate_points=sols, x_init=x0, use_scipy=use_scipy_reference)
                tag = f"smooth_sigma_{sigma:g}_{loss}_mu_{mu:g}"
                plot_histories(hists, f_star=f_star, title_prefix=tag, out_dir=str(out), y_label_norm="gradient norm")

                summary[tag] = {
                    "sigma": float(sigma),
                    "loss": loss,
                    "mu": float(mu),
                    "L0": float(L0),
                    "cond_ata": condition_number_ata(A),
                    "f_star": float(f_star),
                    "GD_final": float(hist_gd["func"][-1]),
                    "FGM_final": float(hist_fg["func"][-1]),
                    "GD_adaptive_final": float(hist_gda["func"][-1]),
                    "FGM_adaptive_final": float(hist_fga["func"][-1]),
                    "GD_best_residual": float(min(_residual_curve(hist_gd, f_star))),
                    "FGM_best_residual": float(min(_residual_curve(hist_fg, f_star))),
                    "GD_adaptive_best_residual": float(min(_residual_curve(hist_gda, f_star))),
                    "FGM_adaptive_best_residual": float(min(_residual_curve(hist_fga, f_star))),
                    "GD_final_grad": float(hist_gd["grad"][-1]),
                    "FGM_final_grad": float(hist_fg["grad"][-1]),
                    "GD_adaptive_final_grad": float(hist_gda["grad"][-1]),
                    "FGM_adaptive_final_grad": float(hist_fga["grad"][-1]),
                    "GD_time": float(hist_gd["time"][-1]),
                    "FGM_time": float(hist_fg["time"][-1]),
                    "GD_adaptive_time": float(hist_gda["time"][-1]),
                    "FGM_adaptive_time": float(hist_fga["time"][-1]),
                    "GD_matvec": int(hist_gd["mat_vec"][-1]),
                    "FGM_matvec": int(hist_fg["mat_vec"][-1]),
                    "GD_adaptive_matvec": int(hist_gda["mat_vec"][-1]),
                    "FGM_adaptive_matvec": int(hist_fga["mat_vec"][-1]),
                    "GD_iterations": int(stat_gd["iterations"]),
                    "FGM_iterations": int(stat_fg["iterations"]),
                    "GD_adaptive_iterations": int(stat_gda["iterations"]),
                    "FGM_adaptive_iterations": int(stat_fga["iterations"]),
                }

    with open(out / "smooth_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_subgradient_experiments(
    n: int = 100,
    m: int = 1000,
    sigma_values: Sequence[float] = (10.0, 1e3, 1e5),
    R_values: Sequence[float] = (1.0, 5.0, 10.0),
    gamma_values: Sequence[float] = (0.01, 0.05),
    n_iters: int = 400,
    seed: int = 6365,
    out_dir: str = "subgradient_results",
) -> Dict[str, Dict[str, float]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, float]] = {}

    for sigma in sigma_values:
        A, b = generate_data(n=n, m=m, sigma=float(sigma), seed=seed)
        x0 = np.zeros(n)

        for loss in ("quadratic", "logistic", "l1"):
            for R in R_values:
                for gamma in gamma_values:
                    x_sg, hist_sg, stat_sg = subgradient_method(A, b, loss, R, x0, n_iters, gamma, normalized=False)
                    x_nsg, hist_nsg, stat_nsg = subgradient_method(A, b, loss, R, x0, n_iters, gamma, normalized=True)

                    tag = f"subgrad_sigma_{sigma:g}_{loss}_R_{R:g}_gamma_{gamma:g}"
                    proxy = float(min(np.min(hist_sg["func"]), np.min(hist_nsg["func"])))
                    plot_histories(
                        {"subgradient": hist_sg, "normalized-subgradient": hist_nsg},
                        f_star=proxy,
                        title_prefix=tag,
                        out_dir=str(out),
                        y_label_norm="subgradient norm",
                    )

                    summary[tag] = {
                        "sigma": float(sigma),
                        "loss": loss,
                        "R": float(R),
                        "gamma": float(gamma),
                        "cond_ata": condition_number_ata(A),
                        "f_star_proxy": proxy,
                        "subgradient_final": float(hist_sg["func"][-1]),
                        "normalized_final": float(hist_nsg["func"][-1]),
                        "subgradient_best_residual": float(min(_residual_curve(hist_sg, proxy))),
                        "normalized_best_residual": float(min(_residual_curve(hist_nsg, proxy))),
                        "subgradient_final_norm": float(hist_sg["grad"][-1]),
                        "normalized_final_norm": float(hist_nsg["grad"][-1]),
                        "subgradient_time": float(hist_sg["time"][-1]),
                        "normalized_time": float(hist_nsg["time"][-1]),
                        "subgradient_matvec": int(hist_sg["mat_vec"][-1]),
                        "normalized_matvec": int(hist_nsg["mat_vec"][-1]),
                        "subgradient_iterations": int(stat_sg["iterations"]),
                        "normalized_iterations": int(stat_nsg["iterations"]),
                    }

    with open(out / "subgradient_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def sanity_check(seed: int = 6365) -> Dict[str, float]:
    n, m, sigma = 20, 80, 100.0
    A, b = generate_data(n=n, m=m, sigma=sigma, seed=seed)
    x0 = np.zeros(n)

    Lq = theory_L(A, "quadratic", 1e-2)
    _, hist_gd, _ = gradient_method(A, b, "quadratic", 1e-2, x0, 10, Lq, adaptive=False)
    _, hist_fg, _ = fast_gradient_method(A, b, "quadratic", 1e-2, x0, 10, Lq, adaptive=True)
    _, hist_sg, _ = subgradient_method(A, b, "l1", 5.0, x0, 10, 0.1, normalized=True)

    return {
        "cond_ata": condition_number_ata(A),
        "gd_final": float(hist_gd["func"][-1]),
        "fgm_final": float(hist_fg["func"][-1]),
        "sg_final": float(hist_sg["func"][-1]),
    }


if __name__ == "__main__":
    print(json.dumps(sanity_check(), indent=2))
