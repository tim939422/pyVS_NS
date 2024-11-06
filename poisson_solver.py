import numpy as np
from scipy.fft import dct, idct

class PoissonSolver:
    def __init__(self, Nx, Ny, dx, dy, BC="P-N"):
        self.BC = BC
        if BC == "P-N":
            kx = np.arange(Nx)[:Nx//2 + 1]
            mwx = 2*(np.cos(2*np.pi*kx/Nx) - 1)/(dx**2)
        elif BC == "N-N":
            kx = np.arange(Nx)
            mwx = 2*(np.cos(np.pi*kx/Nx) - 1)/dx**2

        ky = np.arange(Ny)
        mwy = 2*(np.cos(np.pi * ky / Ny) - 1)/dy**2

        # assemble the Laplacian operator
        # (d^2/dx^2 + d^2/dy^2)phi
        MWX, MWY = np.meshgrid(mwx, mwy, indexing="xy")
        self.laplacian = MWX + MWY
        self.laplacian[0, 0] = 1.0 # [0, 0] mode (mean) will not be solved
        self.inverse_laplacian = 1.0/self.laplacian

    
    def solve(self, f):
        f_hat = self._forward(f)
        phi_hat = self.inverse_laplacian*f_hat
        # Fix mean by setting (0,0) frequency component to 0
        phi_hat[0, 0] = 0
        phi = self._backward(phi_hat)

        return phi

    def _forward(self, u):
        if self.BC == "P-N":
            u_hat = dct(np.fft.rfft(u, axis=1), type=2, axis=0, norm="ortho")
        elif self.BC == "N-N":
            u_hat = dct(dct(u, axis=1, norm="ortho", type=2), type=2, axis=0, norm="ortho")

        return u_hat

    def _backward(self, u_hat):
        if self.BC == "P-N":
            u = np.fft.irfft(idct(u_hat, type=2, norm="ortho", axis=0), axis=1)
        elif self.BC == "N-N":
            u = idct(idct(u_hat, type=2, norm="ortho", axis=0), type=2, norm="ortho", axis=1)

        return u
