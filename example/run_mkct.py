# %%
import numpy as np
import matplotlib.pyplot as plt

from mkct import MKCT_solver

def do_fft(t, y, zero_padding=0, nblocks=20):
    dt = t[1] - t[0]
    if zero_padding > 0:
        y_blockd = np.array_split(y, nblocks)
        y_blockd_avg = np.array([np.mean(y_block) for y_block in y_blockd])
        y_last = y_blockd_avg[-1]
        y_first = y[0]
        rate_est = (np.abs(y_last) - np.abs(y_first)) / (t[-1] - t[0])
        
        zeros = y_last * np.exp(np.arange(1, zero_padding + 1)  * rate_est * dt)    
        y = np.concatenate((y, zeros))
    
    w = np.fft.fftfreq(len(y), t[1] - t[0]) * 2 * np.pi
    w = np.fft.fftshift(w)
    
    Y = np.fft.fft(y)
    Y = np.fft.fftshift(Y) * dt
    return w, Y

        
def main():
    tx, re, im = np.loadtxt("./prop-pol-1.dat", unpack=True)
    C_exact = re + 1j * im  
    C_exact /= C_exact[0]   
    
    re, im = np.loadtxt("./moments.dat", unpack=True)
    Omega_n = re + 1j * im 
    
    re, im = np.loadtxt("./tilde_moments.dat", unpack=True)
    tilde_Omega_n = re + 1j * im
    
    solver = MKCT_solver.init(Omega_n, tilde_Omega_n, rescale=0.01)
    
    # t1, C1 = solver.solve_hardtruncation(tf=50, dt=0.005, kernel_domain='time', conv_domain='frequency')
    # t2, C2 = solver.solve_hardtruncation(tf=50, dt=0.005, kernel_domain='frequency', conv_domain='frequency')
    # t3, C3 = solver.solve_pade(tf=50, dt=0.005, kernel_order=1, pade_order=(6, 6), conv_domain='frequency')
    t4, C4 = solver.solve_pade(tf=50, dt=0.002, kernel_order=1, pade_order=(6, 9), conv_domain='time')
    
    fig = plt.figure(dpi=300)
    gs = fig.add_gridspec(2, 1)
    axs = gs.subplots(sharex=True)
    ax = axs[0]
    # ax.plot(t1, C1.real, label="MKCT (hard, time, freq)")
    # ax.plot(t2, C2.real, label="MKCT (hard, freq, freq)")   
    # ax.plot(t3, C3.real, label="MKCT (Pade, freq)")   
    # ax.plot(t4, C4.real, label="MKCT (Pade, time)") 
    ax.plot(t4, C4.real, label="MKCT") 
    ax.plot(tx, C_exact.real, color='k', label="DEOM", ls='--') 
    ax.set_xlim(-0.5, 50)  
    
    ax = axs[1]
    # ax.plot(t1, C1.imag)  
    # ax.plot(t2, C2.imag)    
    # ax.plot(t3, C3.imag)
    ax.plot(t4, C4.imag)
    ax.plot(tx, C_exact.imag, color='k', ls='--')
    axs[0].legend() 
    
    # Zero padding to imr 
    Npad = C_exact.size * 20
    print(f"{Npad=}")
    wx, Cwx = do_fft(tx, C_exact, Npad)
    w, Cw = do_fft(t4, C4, Npad)
    
    Delta = 2
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(w /Delta, Cw.real, label="MKCT")
    ax.plot(wx/Delta, Cwx.real, color='k', ls='--', label="DEOM")
    L = 1
    ax.set_xlim(1-L, 1+L)
    ax.set_xlabel(r"$\omega/\Delta$")
    ax.set_ylabel(r"$I(\omega)$ (arb. units)")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
    
# %%
