import numpy as np
import matplotlib.pyplot as plt
import h5py
import scienceplots
plt.style.use(["science", "nature"])

# load data
with h5py.File("visu_100000.h5") as f:
    tau_p = 0.5
    mu_p = 0.005
    Ny, Nx = f["p"].shape
    sigma_xx = mu_p/tau_p*(f["C11"][:, :] - 1.0)
    sigma_xy = mu_p/tau_p*f["C12"][:, :]
    sigma_yy = mu_p/tau_p*(f["C22"][:, :] - 1.0)

Lx = 1.0; Ly = 1.0
dx = Lx/Nx; dy = Ly/Ny
x = np.arange(Nx)*dx + 0.5*dx
y = np.arange(Ny)*dy + 0.5*dy

# normalization
U = 1.0
sigma0 = mu_p*U/Lx

x_ref, sigma_xx_ref, sigma_yy_ref, sigma_xy_ref = np.loadtxt("openfoam_results/horizontal_sigma.txt", skiprows=1, delimiter=",", usecols=(0, 3, 4, 6), unpack=True)
fig, ax = plt.subplots(1, 1)

l, = ax.plot(x, sigma_xx[Ny//4*3, :]/sigma0)
ax.plot(x_ref, -sigma_xx_ref/sigma0, "o", color=l.get_color(), markerfacecolor="none", markevery=10)

l, = plt.plot(x, sigma_xy[Ny//4*3, :]/sigma0)
ax.plot(x_ref, -sigma_xy_ref/sigma0, "s", color=l.get_color(), markerfacecolor="none", markevery=10)

l, = plt.plot(x, sigma_yy[Ny//4*3, :]/sigma0)
ax.plot(x_ref, -sigma_yy_ref/sigma0, "<", color=l.get_color(), markerfacecolor="none", markevery=10)

ax.set_xticks([0, 0.5, 1.0])
ax.set_xlabel(r"$x/L$")
ax.set_ylabel(r"$\boldsymbol{\sigma}/\sigma_0$")

fig.savefig("stress_horizontal.jpg", dpi=1000)




