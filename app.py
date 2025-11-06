import time
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm

# --- Optional dependencies: numba and QuantLib ---
try:
    import numba
    njit = numba.njit
except Exception:  # numba not available
    def njit(signature_or_function=None, **kwargs):
        def decorator(func):
            return func
        if callable(signature_or_function):
            return signature_or_function
        return decorator

try:
    import QuantLib as ql  # type: ignore
    QL_AVAILABLE = True
except Exception:
    ql = None
    QL_AVAILABLE = False

st.set_page_config(page_title="Opciones FD Dashboard", layout="wide")

# =====================
# Core numerical methods
# =====================
@njit
def _tridiagonal_solve(a, b, c, d):
    n = len(d)
    cp = np.empty(n - 1)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
    dp[n - 1] = (d[n - 1] - a[n - 2] * dp[n - 2]) / (b[n - 1] - a[n - 2] * cp[n - 2])
    x = np.empty(n)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x

@njit
def _tridiagonal_psor(aL, aD, aU, b, P_exercise, omega, P_guess, tol, max_iter):
    N = len(P_guess)
    P = np.copy(P_guess)
    P_old = np.empty_like(P)
    for k in range(max_iter):
        for i in range(N):
            P_old[i] = P[i]
        sigma = aU[0] * P_old[1]
        P_gs = (b[0] - sigma) / aD[0]
        P_sor = (1.0 - omega) * P_old[0] + omega * P_gs
        P[0] = max(P_sor, P_exercise[0])
        for i in range(1, N - 1):
            sigma = aL[i] * P[i - 1] + aU[i] * P_old[i + 1]
            P_gs = (b[i] - sigma) / aD[i]
            P_sor = (1.0 - omega) * P_old[i] + omega * P_gs
            P[i] = max(P_sor, P_exercise[i])
        sigma = aL[N - 1] * P[N - 2]
        P_gs = (b[N - 1] - sigma) / aD[N - 1]
        P_sor = (1.0 - omega) * P_old[N - 1] + omega * P_gs
        P[N - 1] = max(P_sor, P_exercise[N - 1])
        error = 0.0
        for i in range(N):
            diff = abs(P[i] - P_old[i])
            if diff > error:
                error = diff
        if error < tol:
            return P, k + 1
    return P, max_iter

def _domain_from_L(K, sigma, T, L):
    xmin = math.log(K) - L * sigma * math.sqrt(T)
    xmax = math.log(K) + L * sigma * math.sqrt(T)
    return xmin, xmax

def _estimate_M_for_C(T, sigma, L, N, C):
    """Estimación rápida de M a partir de C (coherente con ΔZ = C·√Δt).
    width_x = xmax - xmin = 2*L*sigma*sqrt(T)
    dx = C*sqrt(dt) = C*sqrt(T/N)
    M ≈ ceil(width_x / dx)
    """
    if T <= 0 or N <= 0 or C <= 0 or sigma <= 0 or L <= 0:
        return math.inf
    width_x = 2.0 * L * sigma * math.sqrt(T)
    dx = C * math.sqrt(T / N) * sigma
    if dx <= 0:
        return math.inf
    return int(math.ceil(width_x / dx))

def _bc_arrays_full(is_call, american, K, r, q, Smin, Smax, tau_vec):
    if is_call:
        left = np.zeros_like(tau_vec)
        right = np.exp(-q * tau_vec) * Smax - K * np.exp(-r * tau_vec)
    else:
        left = K * np.ones_like(tau_vec) if american else K * np.exp(-r * tau_vec)
        right = np.zeros_like(tau_vec)
    return left, right

def solve_pde(K, T, r, sigma, q=0.0, is_call=False, american=True, M=200, N=1000,
              L=6.0, xmin=None, xmax=None, method="implicit", american_method='projection'):
    if xmin is None or xmax is None:
        xmin, xmax = _domain_from_L(K, sigma, T, L)
    x = np.linspace(xmin, xmax, M + 1)
    Z = np.exp(x)
    dx = (xmax - xmin) / M
    dt = T / N
    a = 0.5 * sigma * sigma
    b = r - q - 0.5 * sigma * sigma
    alpha = a * dt / (dx * dx)
    beta = b * dt / (2.0 * dx)
    gamma = r * dt
    U = np.empty((N + 1, M + 1))
    payoff = np.maximum(Z - K, 0.0) if is_call else np.maximum(K - Z, 0.0)
    U[0] = payoff
    tau_vec = np.arange(N + 1) * dt
    Smin = math.exp(xmin)
    Smax = math.exp(xmax)
    left_bc, right_bc = _bc_arrays_full(is_call, american, K, r, q, Smin, Smax, tau_vec)
    U[:, 0] = left_bc
    U[:, -1] = right_bc

    if method == "explicit":
        a_star = (-beta + alpha) / (1.0 + gamma)
        b_star = (1.0 - 2.0 * alpha) / (1.0 + gamma)
        g_star = (beta + alpha) / (1.0 + gamma)
        for n in range(N):
            U[n + 1, 1:-1] = a_star * U[n, 0:-2] + b_star * U[n, 1:-1] + g_star * U[n, 2:]
            if american:
                U[n + 1, 1:-1] = np.maximum(U[n + 1, 1:-1], payoff[1:-1])
    elif method == "implicit":
        lower_coeffs = (beta - alpha) * np.ones(M - 1)
        diag_coeffs = (1.0 + 2.0 * alpha + gamma) * np.ones(M - 1)
        upper_coeffs = (-(beta + alpha)) * np.ones(M - 1)
        aL = lower_coeffs.copy()
        aD = diag_coeffs.copy()
        aU = upper_coeffs.copy()
        aL[0] = 0.0
        aU[-1] = 0.0
        for n in range(N):
            rhs = U[n, 1:-1].copy()
            rhs[0] -= lower_coeffs[0] * U[n + 1, 0]
            rhs[-1] -= upper_coeffs[-1] * U[n + 1, -1]
            if not american:
                U[n + 1, 1:-1] = _tridiagonal_solve(aL, aD, aU, rhs)
            else:
                if american_method == 'projection':
                    U_hold = _tridiagonal_solve(aL, aD, aU, rhs)
                    U[n + 1, 1:-1] = np.maximum(U_hold, payoff[1:-1])
                elif american_method == 'psor':
                    P_guess = U[n, 1:-1]
                    P_ex = payoff[1:-1]
                    U_sol, _iters = _tridiagonal_psor(
                        aL, aD, aU, rhs,
                        P_ex,
                        1.4,  # omega
                        P_guess,
                        1e-8,
                        10000,
                    )
                    U[n + 1, 1:-1] = U_sol
                else:
                    raise ValueError("american_method debe ser 'psor' o 'projection'")
    else:
        raise ValueError("method must be 'explicit' or 'implicit' per Hull's scheme")

    return Z, U[-1]

# --- Pricing helpers ---
def black_scholes_european(S, K, T, r, sigma, q, is_call):
    if T <= 0:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return (S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))
    else:
        return (K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1))


def exact_value(S, K, T, r, sigma, q, is_call):
    """Reference via QuantLib FD if available, else returns None."""
    if not QL_AVAILABLE:
        return None
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    days = max(1, int(round(T * 365)))
    maturity = today + ql.Period(days, ql.Days)

    opt_type = ql.Option.Call if is_call else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(opt_type, K)
    exercise = ql.AmericanExercise(today, maturity)
    option = ql.VanillaOption(payoff, exercise)

    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    spot = ql.SimpleQuote(S)
    r_ts = ql.FlatForward(today, r, day_count)
    q_ts = ql.FlatForward(today, q, day_count)
    vol_ts = ql.BlackConstantVol(today, calendar, sigma, day_count)

    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(spot),
        ql.YieldTermStructureHandle(q_ts),
        ql.YieldTermStructureHandle(r_ts),
        ql.BlackVolTermStructureHandle(vol_ts),
    )

    engine = ql.FdBlackScholesVanillaEngine(process, 2000, 400)
    option.setPricingEngine(engine)
    return float(option.NPV())


# =============
# Experiment run
# =============
@st.cache_data(show_spinner=False)
def run_experiment(K, T, r, sigma, q, S0, is_call, american, N, C_values, methods, L=6.0):
    xmin, xmax = _domain_from_L(K, sigma, T, L)
    dt = T / N

    results = {
        'C_values': list(C_values),
        'explicit': {'price': [], 'time': []},
        'implicit_proj': {'price': [], 'time': []},
        'implicit_psor': {'price': [], 'time': []},
        'bs_european': {'price': [], 'time': []},
    }

    for C in C_values:
        dx = C * sigma *  math.sqrt(dt)
        M = int(math.ceil((xmax - xmin) / dx))

        # Rechazo suave: si M*N supera 1e6, marcamos NaN y continuamos
        if M * N > 50_000_000:
            if 'Explícito' in methods:
                results['explicit']['price'].append(np.nan)
                results['explicit']['time'].append(np.nan)
            if 'Implícito (Proyección)' in methods:
                results['implicit_proj']['price'].append(np.nan)
                results['implicit_proj']['time'].append(np.nan)
            if 'Implícito (PSOR)' in methods:
                results['implicit_psor']['price'].append(np.nan)
                results['implicit_psor']['time'].append(np.nan)
            if 'BS Europeo' in methods:
                price_bs = black_scholes_european(S0, K, T, r, sigma, q, is_call)
                results['bs_european']['price'].append(price_bs)
                results['bs_european']['time'].append(np.nan)
            continue

        if 'Explícito' in methods:
            try:
                t0 = time.time()
                S_vec, V_vec = solve_pde(K, T, r, sigma, q=q, is_call=is_call, american=american,
                                         M=M, N=N, method='explicit')
                exec_t = time.time() - t0
                price = float(np.interp(S0, S_vec, V_vec))
                finite = np.all(np.isfinite(V_vec)) and np.isfinite(price)
                if (not finite) or (price > 10 * K):
                    results['explicit']['price'].append(np.nan)
                    results['explicit']['time'].append(exec_t)
                else:
                    results['explicit']['price'].append(price)
                    results['explicit']['time'].append(exec_t)
            except Exception:
                results['explicit']['price'].append(np.nan)
                results['explicit']['time'].append(np.nan)

        if 'Implícito (Proyección)' in methods:
            t0 = time.time()
            S_vec, V_vec = solve_pde(K, T, r, sigma, q=q, is_call=is_call, american=american,
                                     M=M, N=N, method='implicit', american_method='projection')
            exec_t = time.time() - t0
            price = float(np.interp(S0, S_vec, V_vec))
            results['implicit_proj']['price'].append(price)
            results['implicit_proj']['time'].append(exec_t)

        if 'Implícito (PSOR)' in methods:
            t0 = time.time()
            S_vec, V_vec = solve_pde(K, T, r, sigma, q=q, is_call=is_call, american=american,
                                     M=M, N=N, method='implicit', american_method='psor')
            exec_t = time.time() - t0
            price = float(np.interp(S0, S_vec, V_vec))
            results['implicit_psor']['price'].append(price)
            results['implicit_psor']['time'].append(exec_t)

        if 'BS Europeo' in methods:
            price_bs = black_scholes_european(S0, K, T, r, sigma, q, is_call)
            results['bs_european']['price'].append(price_bs)
            results['bs_european']['time'].append(0.0)

    # Reference price for error plots
    ql_exact = exact_value(S0, K, T, r, sigma, q, is_call)
    ref_price = (ql_exact if ql_exact is not None else (
                 np.nanmean([p for p in results['implicit_psor']['price'] if not np.isnan(p)])
                 if len(results['implicit_psor']['price']) > 0 else
                 np.nan))

    # Compute errors against reference
    for key, label in [('explicit', 'Explícito'), ('implicit_proj', 'Implícito (Proyección)'),
                       ('implicit_psor', 'Implícito (PSOR)'), ('bs_european', 'BS Europeo')]:
        if key in results:
            errs = []
            for p in results[key]['price']:
                if p is None or (isinstance(p, float) and math.isnan(p)):
                    errs.append(np.nan)
                else:
                    errs.append(abs(p - ref_price))
            results[key]['error'] = errs

    return results, ref_price


def build_figure(results, reference_price, title_suffix=""):
    x_axis_data = results['C_values']
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 16), sharex=True)
    fig.suptitle(f"Análisis Comparativo vs. C {title_suffix}")

    def aplot(ax, y, *args, **kwargs):
        try:
            if y is not None and len(y) == len(x_axis_data) and len(y) > 0:
                ax.plot(x_axis_data, y, *args, **kwargs)
        except Exception:
            pass

    # 1) Prices
    aplot(ax1, results['explicit']['price'], 'o-', label='Explícito')
    aplot(ax1, results['implicit_proj']['price'], 's-', label='Implícito (Proyección)')
    aplot(ax1, results['implicit_psor']['price'], '^-', label='Implícito (PSOR)')
    if len(results['bs_european']['price']) > 0 and len(results['bs_european']['price']) == len(x_axis_data):
        ax1.axhline(y=results['bs_european']['price'][0], color='grey', linestyle='--', label='BS Europeo (Ref)')
    if reference_price is not None and not (isinstance(reference_price, float) and math.isnan(reference_price)):
        ax1.axhline(y=reference_price, color='k', linestyle=':', label='Referencia (QL o PSOR)')
    ax1.set_ylabel('Precio de la Opción ($)')
    ax1.set_title('Precio Obtenido vs. C')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 2) Errors
    aplot(ax2, results['explicit'].get('error'), 'o-', label='Explícito')
    aplot(ax2, results['implicit_proj'].get('error'), 's-', label='Implícito (Proyección)')
    aplot(ax2, results['implicit_psor'].get('error'), '^-', label='Implícito (PSOR)')
    ax2.set_ylabel('Error Absoluto (vs. Ref)')
    ax2.set_title('Error vs. C')
    ax2.set_yscale('log')
    ax2.grid(True, which='both', linestyle='--', alpha=0.6)
    ax2.legend()

    # 3) Times
    aplot(ax3, results['explicit']['time'], 'o-', label='Explícito')
    aplot(ax3, results['implicit_proj']['time'], 's-', label='Implícito (Proyección)')
    aplot(ax3, results['implicit_psor']['time'], '^-', label='Implícito (PSOR)')
    ax3.set_xlabel('Factor C (ΔZ = C · √Δt)')
    ax3.set_ylabel('Tiempo de Ejecución (s)')
    ax3.set_title('Velocidad vs. C')
    ax3.set_yscale('log')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def build_separate_figures(results, reference_price, xlabel_text):
    x_axis_data = results['C_values']
    x_len = len(x_axis_data)

    def pplot(y, *args, **kwargs):
        try:
            if y is not None and len(y) == x_len and x_len > 0:
                plt.plot(x_axis_data, y, *args, **kwargs)
        except Exception:
            pass

    # Precio
    fig1 = plt.figure(figsize=(9, 5))
    pplot(results['explicit']['price'], 'o-', label='Explícito')
    pplot(results['implicit_proj']['price'], 's-', label='Implícito (Proyección)')
    pplot(results['implicit_psor']['price'], '^-', label='Implícito (PSOR)')
    if len(results['bs_european']['price']) > 0:
        plt.axhline(y=results['bs_european']['price'][0], color='grey', linestyle='--', label='BS Europeo (Referencia)')
    if reference_price is not None and not (isinstance(reference_price, float) and math.isnan(reference_price)):
        plt.axhline(y=reference_price, color='k', linestyle=':', label='Referencia (QL o PSOR)')
    plt.ylabel('Precio de la Opción ($)')
    plt.xlabel(xlabel_text)
    plt.title('Análisis Comparativo: Precio Obtenido vs. C')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Error
    fig2 = plt.figure(figsize=(9, 5))
    pplot(results['explicit'].get('error'), 'o-', label='Explícito')
    pplot(results['implicit_proj'].get('error'), 's-', label='Implícito (Proyección)')
    pplot(results['implicit_psor'].get('error'), '^-', label='Implícito (PSOR)')
    plt.ylabel('Error Absoluto (vs. Referencia)')
    plt.xlabel(xlabel_text)
    plt.title('Análisis Comparativo: Error vs. C')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Tiempo
    fig3 = plt.figure(figsize=(9, 5))
    pplot(results['explicit']['time'], 'o-', label='Explícito')
    pplot(results['implicit_proj']['time'], 's-', label='Implícito (Proyección)')
    pplot(results['implicit_psor']['time'], '^-', label='Implícito (PSOR)')
    plt.xlabel(xlabel_text)
    plt.ylabel('Tiempo de Ejecución (s)')
    plt.title('Análisis Comparativo: Velocidad vs. C')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    return fig1, fig2, fig3

# =====================
# UI
# =====================
st.title("Dashboard de Precios de Opciones (FD)")
st.caption("Explorá métodos Explícito e Implícito (Proyección/PSOR) y comparalos con BS Europeo y referencia QuantLib si está disponible.")

with st.sidebar:
    st.header("Modo")
    page = st.radio("Elegí sección:", ["Ejemplos", "Parámetros personalizados"])  # type: ignore

    
METHODS_DEFAULT = ('Explícito', 'Implícito (Proyección)', 'Implícito (PSOR)', 'BS Europeo')
# ------------------
# Predefined examples
# ------------------
if page == "Ejemplos":
    st.subheader("Ejemplos preconfigurados (imágenes pre-generadas)")

    import os
    IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images', 'examples')
    os.makedirs(IMAGES_DIR, exist_ok=True)

    examples = {
        # Base y variaciones a partir del base
        "base": dict(title="Base (Put Americano)", K=100.0, T=1.0, r=0.05, sigma=0.30, q=0.01, S0=100.0,
                      is_call=False, american=True, N=1000, C_values=[0.1, 0.25, 0.5, 1.0, 3**(1/2), 2.0, 3.0]),
        "base_n100": dict(title="Base variando N=100", K=100.0, T=1.0, r=0.05, sigma=0.30, q=0.01, S0=100.0,
                           is_call=False, american=True, N=100, C_values=[0.1, 0.25, 0.5, 1.0, 3**(1/2), 2.0, 3.0]),
        "base_n10000": dict(title="Base N=10000", K=100.0, T=1.0, r=0.05, sigma=0.30, q=0.01, S0=100.0,
                                is_call=False, american=True, N=10000, C_values=[0.25, 0.5, 1.0, 3**(1/2), 2.0, 3.0]),
        "base_s0_120": dict(title="Base variando S0=120", K=100.0, T=1.0, r=0.05, sigma=0.30, q=0.01, S0=120.0,
                             is_call=False, american=True, N=1000, C_values=[0.1, 0.25, 0.5, 1.0, 3**(1/2), 2.0, 3.0]),
        "base_sigma_035": dict(title="Base variando σ=0.35", K=100.0, T=1.0, r=0.05, sigma=0.5, q=0.01, S0=100.0,
                                is_call=False, american=True, N=1000, C_values=[0.1, 0.25, 0.5, 1.0, 3**(1/2), 2.0, 3.0]),
    }

    def image_paths(slug):
        return (
            os.path.join(IMAGES_DIR, f"{slug}_price.png"),
            os.path.join(IMAGES_DIR, f"{slug}_error.png"),
            os.path.join(IMAGES_DIR, f"{slug}_time.png"),
        )

    def ensure_images_for_example(slug, p):
        p_price, p_error, p_time = image_paths(slug)
        need = not (os.path.exists(p_price) and os.path.exists(p_error) and os.path.exists(p_time))
        if need:
            with st.spinner(f"Generando imágenes para {p['title']}…"):
                # Métodos por defecto para ejemplos
                default_methods = ['Explícito', 'Implícito (Proyección)', 'Implícito (PSOR)', 'BS Europeo']
                results, ref = run_experiment(
                    p['K'], p['T'], p['r'], p['sigma'], p['q'], p['S0'],
                    p['is_call'], p['american'], p['N'], tuple(p['C_values']), tuple(default_methods)
                )
                xlabel = 'Relación C (ΔZ = C · σ · √Δt)'
                fig1, fig2, fig3 = build_separate_figures(results, ref, xlabel)
                fig1.savefig(p_price, dpi=150)
                plt.close(fig1)
                fig2.savefig(p_error, dpi=150)
                plt.close(fig2)
                fig3.savefig(p_time, dpi=150)
                plt.close(fig3)
        return image_paths(slug)

    # Mostrar todas las imágenes de todos los ejemplos a la vez en filas
    for slug, p in examples.items():
        p_price, p_error, p_time = ensure_images_for_example(slug, p)
        st.markdown(f"#### {p['title']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(p_price, caption='Precio vs C', use_container_width=True)
        with col2:
            st.image(p_error, caption='Error vs C', use_container_width=True)
        with col3:
            st.image(p_time, caption='Velocidad vs C', use_container_width=True)
        # Parámetros pequeños debajo
        st.caption(
            f"K={p['K']}, S0={p['S0']}, r={p['r']}, q={p['q']}, σ={p['sigma']}, T={p['T']}, N={p['N']}, C={p['C_values']}"
        )
        st.divider()

# ------------------
# Custom parameters
# ------------------
else:
    st.subheader("Generar gráfico con tus parámetros")

    colA, colB, colC = st.columns(3)
    with colA:
        K = st.number_input("Strike K", value=100.0, min_value=1.0, step=1.0)
        r = st.number_input("Tasa libre de riesgo r", value=0.05, min_value=0.0, max_value=1.0, step=0.01, format="%.4f")
        q = st.number_input("Dividendo continuo q", value=0.01, min_value=0.0, max_value=1.0, step=0.01, format="%.4f")
    with colB:
        S0 = st.number_input("Spot S0", value=105.0, min_value=0.01, step=1.0)
        sigma = st.number_input("Volatilidad σ", value=0.5, min_value=0.30, max_value=5.0, step=0.01, format="%.4f")
        T = st.number_input("Tiempo a vencimiento T (años)", value=1.0, min_value=0.001, max_value=10.0, step=0.1)
    with colC:
        style = st.selectbox("Tipo", ["Put Americano", "Call Americano", "Put Europeo", "Call Europeo"])
        is_call = "Call" in style
        american = "Americano" in style
        N = st.number_input("N (pasos temporales)", value=200, min_value=50, max_value=300, step=10)

    st.markdown("#### Valores de C (ΔZ / √Δt)")
    c1, c2 = st.columns(2)
    with c1:
        c_min = st.number_input("C mínimo", value=0.5, min_value=0.5, step=0.025, format="%.3f")
        c_mid = st.number_input("C intermedio", value=1.0, min_value=0.5, step=0.025, format="%.3f")
    with c2:
        c_mid2 = st.number_input("C intermedio 2", value=3**(1/2), min_value=0.5, step=0.025, format="%.3f")
        c_max = st.number_input("C máximo", value=2.0, min_value=0.5, step=0.025, format="%.3f")

    c_list = [c_min, c_mid, c_mid2, c_max]
    c_list = sorted(set([round(float(c), 4) for c in c_list]))
    st.write("C evaluados:", c_list)

    # Validación anticipada M*N <= 1e6
    invalid = []
    for C in c_list:
        M_est = _estimate_M_for_C(T, float(sigma), 6.0, int(N), float(C))
        if M_est == math.inf or M_est * int(N) > 50_000_000:
            invalid.append((C, M_est))
    if invalid:
        msg_lines = [f"- C={c}: M≈{(m if m!=math.inf else '∞')} → M*N>50_000_000" for c, m in invalid]
        st.error("Las opciones elegidas exceden el límite de malla (M*N ≤ 10.000.000). Ajustá N o aumentá C.\n" + "\n".join(msg_lines))

    if st.button("Calcular con mis parámetros"):
        if invalid:
            st.stop()
        with st.spinner("Calculando…"):
            results, ref = run_experiment(
                K, T, r, sigma, q, S0, is_call, american, int(N), tuple(c_list), METHODS_DEFAULT
            )
        xlabel = 'Relación C (ΔZ = C · √Δt)'
        fig1, fig2, fig3 = build_separate_figures(results, ref, xlabel)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.pyplot(fig1, clear_figure=False)
        with col2:
            st.pyplot(fig2, clear_figure=False)
        with col3:
            st.pyplot(fig3, clear_figure=False)

st.divider()
# st.caption("Sugerencia: habilitá PSOR para evaluar la cota inferior (ejercicio temprano). El método Explícito puede divergir para C grandes.")
