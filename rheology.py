import numpy as np
from utils import cell2node

class Rheology:
    def __init__(self, Ntot_x, Ntot_y):
        self.buffer = np.zeros((Ntot_y, Ntot_x))

    def calculate_newtonian_stress(self, mu_s,
                                   dudx, dudy, dvdx, dvdy,
                                   sigma_xx, sigma_xy, sigma_yy):
        sigma_xx = 2*mu_s*dudx
        sigma_xy = mu_s*(dvdx + dudy)
        sigma_yy = 2*mu_s*dvdy

        return sigma_xx, sigma_xy, sigma_yy

    def calculate_Oldroyd_B_stress(self, mu_s, mu_p, tau_p,
                                   Cmat, dudx, dudy, dvdx, dvdy,
                                   sigma_xx, sigma_xy, sigma_yy):
        sigma_xx, sigma_xy, sigma_yy = self.calculate_newtonian_stress(mu_s,
                                                                       dudx, dudy, dvdx, dvdy,
                                                                       sigma_xx, sigma_xy, sigma_yy)

        # compute C_12 at node for shear stress
        self.buffer = cell2node(Cmat[1, :, :], self.buffer)

        factor = mu_p/tau_p

        sigma_xx += factor*(Cmat[0, :, :] - 1.0)
        sigma_xy += factor*self.buffer
        sigma_yy += factor*(Cmat[3, :, :] - 1.0)

        return sigma_xx, sigma_xy, sigma_yy


        
