# %%
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt

from mkct.core import autocorr_convolution
from mkct.core import get_Kn_initial
from mkct.core import get_K_propagator
from mkct.core import get_K1s
from mkct.core import pade_approx_Knt_func
from mkct.core import tilde_Omega_from_Omega_recursive

from dataclasses import dataclass
from typing import List, Tuple, Callable, Union
import time
import math

@dataclass
class MKCT_solver:
    Omega_n: NDArray[np.complex128]
    tilde_Omega_n: NDArray[np.complex128]
    Kn_0: NDArray[np.complex128]
    L_K: NDArray[np.complex128]
    rescale: float
    t: NDArray[np.float64] = None
    K1t: NDArray[np.complex128] = None

    @classmethod
    def init(
        cls,
        Omega_n: NDArray[np.complex128],
        rescale: float=1.0,
    ) -> 'MKCT_solver':
        # compute the initial condition
        Kn_0 = get_Kn_initial(Omega_n)

        # Compute the maximum possible order of the memory kernel
        # propagator matrix
        L_K = get_K_propagator(Omega_n)

        tilde_Omega_n = tilde_Omega_from_Omega_recursive(Omega_n)

        return cls(Omega_n, tilde_Omega_n, Kn_0, L_K, rescale)

    @property
    def max_kernel_order(self) -> int:
        return len(self.Kn_0)

    @property
    def max_deriv_order(self) -> int:
        return len(self.tilde_Omega_n) - 1

    @staticmethod
    def tab_str(
        string: str,
        N: int=69,
        border: str='!',
        tab: str='=',
        fill_tab: str=' ',
    ) -> str:
        if string == "":
            return border + tab * (N-2) + border
        else:
            left_margin = (N - len(string)) // 2
            right_margin = N - len(string) - left_margin
            return border + fill_tab * left_margin + string + fill_tab * right_margin + border + "\n"

    def give_size_hint_str(self) -> str:
        m_max = self.max_deriv_order
        n_plus_m_plus_1 = len(self.Omega_n)
        n_max = n_plus_m_plus_1 - m_max - 1

        hint_str = self.tab_str(" Size Hint ", fill_tab='=')
        hint_str += self.tab_str(f"Given {n_plus_m_plus_1} Ω_n, and {m_max} tilde_Ω_n,")
        # hint_str += self.tab_str(f"You can compute the Pade approximant up to order {n_max},")
        # hint_str += self.tab_str(f"with a maximum Pade approximant order of {m_max}.")
        # hint_str += self.tab_str(f"Highest order kernel which can use all the tilde_Ω_n is {m_max}.")
        hint_str += self.tab_str(f"You can evaluate K_1(t) to K_{n_max}(t) with {m_max} derivatives,")
        hint_str += self.tab_str(f"or K_n(t), n < {n_plus_m_plus_1-1}, with less than {m_max} derivatives.")
        hint_str += self.tab_str("=", fill_tab='=')
        return hint_str

    def __str__(self) -> str:
        cls_name = self.__class__.__name__
        cls_str = f"{cls_name}({self.Omega_n.size} Ω_n, {self.tilde_Omega_n.size} tilde_Ω_n, rescale={self.rescale})\n"
        size_hint = self.give_size_hint_str()
        return cls_str + "\n" + size_hint

    def solve_hardtruncation(
        self,
        tf: float,
        dt: float,
        kernel_domain: str='time',
        conv_domain: str='time',
    ) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
        t = np.arange(0, tf+dt, dt)
        t_rescaled = t / self.rescale
        dt_rescaled = dt / self.rescale

        # First, compute the memory kernel
        if kernel_domain == 'time':
            # compute the memory kernel via expm_multiply
            Kn_t = expm_multiply(self.L_K, self.Kn_0, t_rescaled[0], t_rescaled[-1], t_rescaled.size, endpoint=True)
            K1_t = Kn_t[:, 0]
        elif kernel_domain == 'frequency':
            # compute the memory kernel in frequency domain
            K1_s_func = get_K1s(Omega_n=self.Omega_n, Kn_0=self.Kn_0)
            s = 1.j * 2 * np.pi * np.fft.fftfreq(t.size, d=dt_rescaled)
            K1_s = K1_s_func(s)

            # compute the inverse FFT to get the memory kernel in time domain
            K1_t = np.fft.ifft(K1_s) / dt_rescaled
        else:
            raise ValueError(
                """Invalid domain for the memory kernel."""
                """Choose either 'time' or 'frequency'."""
                """Got: {kernel_domain}"""
            )

        # Next, compute the convolution
        _, C = autocorr_convolution(
            Omega_1=self.Omega_n[0],
            K1_func=(t_rescaled, K1_t),
            tf=tf/self.rescale,
            dt=dt/self.rescale,
            domain=conv_domain,
        )

        self.K1t = K1_t
        return t, C

    def solve_pade(
        self,
        tf: float,
        dt: float,
        kernel_order: int,
        pade_order: Union[int, Tuple[int, int]],
        conv_domain: str='time',
        post_ifft_smooth: int = 10,
    ) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
        m = sum(pade_order) if isinstance(pade_order, tuple) else pade_order
        assert m <= self.max_deriv_order, f"With {self.max_deriv_order} tilde_Ω_n, the maximum Pade order is {self.max_deriv_order}."

        if kernel_order == 1:
            K1_func = pade_approx_Knt_func(
                Omega_n=self.Omega_n,
                tilde_Omega_n=self.tilde_Omega_n,
                n=1,
                m=pade_order,
            )
            t_rescaled = np.arange(0, tf+dt, dt) / self.rescale
            dt_rescaled = dt / self.rescale
            s = 1.j * 2 * np.pi * np.fft.fftfreq(t_rescaled.size, d=dt_rescaled)
            sp = s
            K1_t = K1_func(t_rescaled)
            K1_s = np.fft.fft(K1_t) * dt_rescaled
        else:
            Km_func, p, q = pade_approx_Knt_func(
                Omega_n=self.Omega_n,
                tilde_Omega_n=self.tilde_Omega_n,
                n=kernel_order,
                m=pade_order,
                return_pq=True,
            )
            t_rescaled = np.arange(0, tf+dt, dt) / self.rescale
            dt_rescaled = dt / self.rescale
            Km_t = Km_func(t_rescaled)
            w = np.fft.fftshift(np.fft.fftfreq(t_rescaled.size, d=dt_rescaled)) * 2 * np.pi
            s = 1.j * w
            sp = s

            Km_s = np.fft.fftshift(np.fft.fft(Km_t)) * dt_rescaled
            Km_s_func = lambda ss: np.interp(ss.imag, s.imag, Km_s)

            Km_s_interp = Km_s_func(s)

            Omega_n_in = self.Omega_n[:kernel_order]
            Kn_0_in = self.Kn_0[:kernel_order-1]
            K1_s = get_K1s(Omega_n=Omega_n_in, Kn_0=Kn_0_in, Kn_plus1=Km_s_func)(s)
            K1_t = np.fft.ifft(np.fft.ifftshift(K1_s)) / dt_rescaled
            K1_func = lambda t: np.interp(t, t_rescaled, K1_t)

        # fig = plt.figure(dpi=300)
        # ax = fig.add_subplot(111)
        # # K1_sp = np.fft.fftshift(K1_s)

        # ax.plot(sp.imag, K1_s.real, label="real")
        # ax.plot(sp.imag, K1_s.imag, label="imag")
        # L = 100
        # ax.set_xlim(-L*self.rescale, L*self.rescale)
        # ax.set_xlabel("w")
        # ax.set_ylabel("K1(w)")
        # ax.legend()
        # plt.show()


            # raise NotImplementedError(
            #     "Solving for higher order kernels using Pade approximants is not implemented yet."
            # )

        _, C1 = autocorr_convolution(
            Omega_1=self.Omega_n[0],
            K1_func=K1_func,
            tf=tf/self.rescale,
            dt=dt/self.rescale,
            domain=conv_domain,
        )

        # if conv_domain == "frequency":
        #     C1[:post_ifft_smooth] = np.mean(C1[:post_ifft_smooth])
        #     C1[-post_ifft_smooth:] = np.mean(C1[-post_ifft_smooth:])

        t = np.arange(0, tf+dt, dt)
        self.t = t
        self.K1t = K1_func(t/self.rescale)
        return t, C1

# %%
