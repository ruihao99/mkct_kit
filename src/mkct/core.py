import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import pade
from scipy.integrate import simpson
import sympy as sp
import mpmath as mp

import math
import warnings
from typing import Tuple, Union, Callable, Optional

# Define types
CvalFunc = Callable[[float], complex]

# Convolution related functions
def autocorr_from_K1_freq(
    Omega_1: complex,
    s_vals: NDArray[np.complex128],
    K1_s: NDArray[np.complex128],
    eta: float = 1E-5,
) -> NDArray[np.complex128]:
    """ The laplace transform of the autocorrelation convolution equation is:
        C_AA(s) = C_{AA}(0) / (s - Omega_1 - K1(s))
    we will always assume C_{AA}(0) = 1
    """
    return 1.0 / (s_vals - Omega_1 - K1_s + eta)


def autocorr_convolution_freq(
    t: NDArray[np.float64],
    K1t: NDArray[np.complex128],
    Omega_1: complex,
) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
    # The frequency grid
    s_vals = 1.j * 2 * np.pi * np.fft.fftfreq(len(t), d=t[1] - t[0])
    s_vals = np.fft.fftshift(s_vals)

    # Compute the Laplace transform of the kernel via FFT
    K_s = np.fft.fft(K1t) * (t[1] - t[0])
    K_s = np.fft.fftshift(K_s)

    # Solve for the Laplace transform of the autocorrelation function
    C_s = autocorr_from_K1_freq(Omega_1, s_vals, K_s)

    # Inverse FFT to get the autocorrelation function
    C_t = np.fft.ifft(np.fft.ifftshift(C_s)) / (t[1] - t[0])

    return t, C_t


def autocorr_from_K1_time1(
    t: NDArray[np.float64],
    K1t: NDArray[np.complex128],
    Omega_1: complex,
) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
    # Number of time steps
    n_steps = len(t)
    dt = t[1] - t[0]
    dt_half = 0.5 * dt

    # allocate the results
    C_t = np.zeros(n_steps, dtype=np.complex128)
    conv_integral = np.zeros(n_steps, dtype=np.complex128)
    C_t[0] = 1.0

    # use the simpson rule to compute the convolution integral
    for i in range(1, n_steps):
        conv_integral[i] = simpson(C_t[:i] * np.flip(K1t[:i]), x=t[:i])
        C_t[i] = C_t[i-1] + dt * (Omega_1 * C_t[i-1] + conv_integral[i-1])
    return t, C_t


def autocorr_convolution(
    Omega_1: complex,
    K1_func: Union[CvalFunc, Tuple[NDArray[np.float64], NDArray[np.float64]]],
    tf: float,
    dt: float,
    domain: str = 'time',
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """method to calculate the autocorrelation function via memory kernel convolution:
        C_AA(t) = Omega_1 C_{AA}(t) + int_0^t dt' K1(t-t') C_{AA}(t')
    here:
    - Omega_1 is first order moment of the correlation function
    - K1 is the First order memory kernel

    Args:
        Omega_1 (complex): first order moment of the correlation function
        K1_func (Union[CvalFunc, Tuple[NDArray[np.float64], NDArray[np.float64]]]): First order memory kernel
        tf (float): final time
        dt (float): time step
        domain (str, optional): choose to solve the convolution in 'time' or 'frequency' domain. Defaults to 'time'.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]: (time, autocorrelation)
    """
    # Step 0: Define the output time grid
    t = np.arange(0, tf+dt, dt)

    # First parse the kernel function
    # two cases:
    # 1. K1 is a function
    # 2. K1 is given as samples (t_mem, K1(t_mem))
    if callable(K1_func):
        t_mem = np.copy(t)
        K1_mem = K1_func(t)
    else:
        try:
            t_mem, K1_mem = K1_func
        except:
            raise ValueError("K1_func should be a function or a tuple of (t_mem, K1_mem)")

    # If the user wishes to solve in the frequency domain
    if domain == 'frequency':
        return autocorr_convolution_freq(t, K1_mem, Omega_1)
    elif domain == 'time':
        # Does the memory time grid match the output grid?
        if t.size == t_mem.size:
            # The grid sizes match
            if np.allclose(t, t_mem):
                # The grids are the same
                # return autocorr_from_K1_time1(t, K1_mem, Omega_1)
                return autocorr_from_K1_time1(t, K1_mem, Omega_1)

        # reach here, means the grids do not match
        # There are better ways to use a shorter memory kernel
        # but for now, we will just interpolate the memory kernel
        warnings.warn(
            """ The memory kernel grid does not match the output grid. """
            """ Interpolating the memory kernel to match the output grid. """
            """ This may lead to inaccuracies. """
        )


        K1_interp = np.interp(t, t_mem, K1_mem)
        return autocorr_from_K1_time1(t, K1_interp, Omega_1)
    else:
        raise ValueError("domain should be either 'time' or 'frequency'")


# MKCT related functions
def get_Kn_initial(Omega_n: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Compute the initial condition terms for the memory kernel."""
    return Omega_n[1:] - Omega_n[0] * Omega_n[:-1]


def get_K_propagator(
    Omega_n: NDArray[np.complex128],
    trunc_order: int = None
) -> NDArray[np.complex128]:
    """Compute the propagator for the memory kernel."""
    if trunc_order is None:
        trunc_order = len(Omega_n) - 1
    elif isinstance(trunc_order, int):
        if trunc_order > len(Omega_n) - 1:
            raise ValueError("trunc_order should be less than the number of frequencies")
    else:
        raise ValueError(f"trunc_order should be an integer or None. Got {trunc_order=}")

    L_K = np.zeros((trunc_order, trunc_order), dtype=np.complex128)

    for irow in range(trunc_order):
        L_K[irow, 0] += (-Omega_n[irow])
        if irow < trunc_order - 1:
            L_K[irow, irow+1] += 1.0 + 0.j
    return L_K

def get_K1s(
    Omega_n: NDArray[np.complex128],
    Kn_0: NDArray[np.complex128],
    Kn_plus1: Optional[CvalFunc] = None
) -> CvalFunc:
    # use sympy to create the expression
    s = sp.symbols('s')

    n_plus1 = len(Omega_n)
    n = len(Kn_0)

    if n_plus1 != n + 1:
        # if not matching, we need to truncate Omega_n or Kn_0
        if n_plus1 > n + 1:
            Omega_n = Omega_n[:n+1]
            n_plus1 = n + 1
        else:
            Kn_0 = Kn_0[:n_plus1-1]
            n = n_plus1 - 1
    m = n + 1


    num = 0
    den = 0

    num_expr = ""
    for i in range(1, m):
        num += Kn_0[i-1] * s**(m-1-i)
        nn = '{' + str(i) + '}'
        od = '{' + str(m-1-i) + '}'
        num_expr += f"K_{nn}(0) * s**{od} + "
    num_expr = num_expr[:-2]

    # The denominator
    den = s**(m-1)
    den_expr = "s**" + '{' + str(m-1) + '}'
    for i in range(1, m):
        den += Omega_n[i-1] * s**(m-1-i)
        den_expr += f" + Omega_{i} * s**" + '{' + str(m-1-i) + '}'

    # print(f"num = {num_expr}")
    # print(f"den = {den_expr}")

    # The final K1(s)
    K1_s = num / den

    # use lambdify to create the function

    if Kn_plus1 is None:
        return sp.lambdify(s, K1_s, 'numpy')
    elif callable(Kn_plus1):
        num_func = sp.lambdify(s, num, 'numpy')
        den_func = sp.lambdify(s, den, 'numpy')
        return lambda s: (Kn_plus1(s) + num_func(s)) / den_func(s)
    else:
        raise ValueError("Kn_plus1 should be a function or None")

def tilde_Omega_from_Omega_recursive(
    Omega_n: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    tilde_Omega_n = np.zeros_like(Omega_n)
    tilde_Omega_n[0] = Omega_n[0]
    for m in range(1, Omega_n.size):
        tilde_Omega_n[m] = Omega_n[m+1-1]
        for k in range(1, m+1):
            tilde_Omega_n[m] -= tilde_Omega_n[m-k] * Omega_n[k-1]
    return tilde_Omega_n


def pade_approx_Knt_func(
    Omega_n: NDArray[np.complex128],
    tilde_Omega_n: NDArray[np.complex128],
    n: int,
    m: Union[int, Tuple[int, int]],
    return_pq: bool = False,
) -> CvalFunc:
    """Use Pade approximation to compute the n-th order memory kernel.
        - n: order of the memory kernel
        - m: maximum derivatives of Kn(t) to use in the approximation
    If m is provided as an integer, the Pade approximant will be of order (m - m//2-1,  m//2+1).
    If m is provided as a tuple, the Pade approximant will be of order (m[0], m[1]).

    Args:
        Omega_n (NDArray[np.complex128]): higher order moments of the correlation function
        tilde_Omega_n (NDArray[np.complex128]): higher order of auxiliary moments
        n (int): order of the memory kernel
        m (Union[int, Tuple[int, int]]): maximum derivatives of Kn(t) to use in the approximation

    Returns:
        CvalFunc: Pade approximant of the n-th order memory kernel
    """
    # First assert that tilde_Omega_0 == Omega_1
    if tilde_Omega_n[0] != Omega_n[0]:
        raise ValueError("tilde_Omega_n[0] should be equal to Omega_n[1]")

    if isinstance(m, int):
        m1 = m2 = None
    elif isinstance(m, tuple):
        m1, m2 = m
        m = m1 + m2 + 1
    else:
        raise ValueError("m should be an integer or a tuple of integers")

    # assert that the length of Omega_n and tilde_Omega_n
    # suffice to compute Knt with order m and n
    if len(Omega_n) < m + n + 1:
        raise ValueError("Omega_n should have at least m + n + 1 elements")


    # Then compute the initial value of Kn(t)
    # Kn_0 = Omega_{n+1} - Omega_1 * Omega_n
    Kn_0 = Omega_n[n+1-1] - Omega_n[0] * Omega_n[n-1]

    # allocate the memory for the polynomial
    Kn_poly = np.zeros(m, dtype=np.complex128)

    # evaluate the taylor series polynomial
    for i in range(m):
        if i == 0:
            Kn_poly[i] = Kn_0
        else:
            Kn_poly[i] = Omega_n[n+i+1-1]
            for j in range(i+1):
                Kn_poly[i] -= tilde_Omega_n[i-j] * Omega_n[n+j-1]

    # Scale the polynomial by the factorial
    for i in range(m):
        Kn_poly[i] /= math.factorial(i)

    # compute the Pade approximant
    if m2 is None:
        p, q = pade(Kn_poly, m//2+1)
    else:
        p, q = pade(Kn_poly, m2, m1)

    if return_pq:
        return lambda t: p(t) / q(t), p, q
    else:
        return lambda t: p(t) / q(t)

# def vectorize(mp_math_func: Callable) -> Callable:
#     """Vectorize a mpmath function to work with numpy arrays."""
#     def vectorized_func(x: NDArray[np.complex128]) -> NDArray[np.complex128]:
#         if np.isscalar(x):
#             # If the input is a scalar, just call the mpmath function
#             return np.complex128(mp_math_func(x))

#         # Convert the input to a list of mpmath.mpf objects
#         x_mpf = [mp.mpc(val.real, val.imag) for val in x]
#         # Call the mpmath function
#         result_mpf = [mp_math_func(x) for x in x_mpf]
#         # Convert the result back to a numpy array
#         return np.array(result_mpf, dtype=np.complex128)
#     return vectorized_func


# def rational_fraction_laplace(p, q):
#     # Define the symbols
#     t, s = sp.symbols('t s')

#     # Evaluate the polynomail at t
#     p_t_expr = sum(p[i] * t**(p.order - i) for i in range(p.order+1))
#     q_t_expr = sum(q[i] * t**(q.order - i) for i in range(q.order+1))
#     f_t = sp.expand(p_t_expr / q_t_expr)
#     print(f"Done evaluating the polynomial")
#     print(f"f(t) = {f_t}")

#     # Compute the Laplace transform
#     f_s = sp.laplace_transform(f_t, t, s, noconds=True)
#     print(f"Done laplace transform")
#     print(f"f(s) = {f_s}")

#     # lambdify the result with mpmath
#     f_s_func = sp.lambdify(s, f_s, 'mpmath')

#     # wrap the mpmath function as a numpy vectorized function
#     # argument type is np.complex128
#     # f_s_func = np.vectorize(lambda s: np.complex128(f_s_func), signature='(n)->(n)')
#     return vectorize(f_s_func)


