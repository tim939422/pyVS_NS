import numpy as np
from utils import node2cell


class CTensor:
    def __init__(self, tau_p, dx, dy,
                 Ntot_x, Ntot_y,
                 ib, ie, jb, je):
        self.tau_p = tau_p
        self.dx = dx
        self.dy = dy

        self.ib = ib; self.ie = ie; self.jb = jb; self.je = je
        self.buffer = np.zeros((Ntot_y, Ntot_x))

    def calculate_RHS(self, Cmat,
                      u, v,
                      dudx, dudy, dvdx, dvdy,
                      rhs_Cmat):
        rhs_Cmat = self._add_spring_force(Cmat, rhs_Cmat)
        rhs_Cmat = self._add_convective(Cmat, u, v, rhs_Cmat)
        rhs_Cmat = self._add_stretched(Cmat, dudx, dudy, dvdx, dvdy, rhs_Cmat)
        return rhs_Cmat
    
    # must be call first !
    def _add_spring_force(self, Cmat, rhs_Cmat):
        rhs_Cmat = -Cmat/self.tau_p
        # extract I from diagonal
        rhs_Cmat[0, :, :] += 1/self.tau_p
        rhs_Cmat[3, :, :] += 1/self.tau_p
        rhs_Cmat[5, :, :] += 1/self.tau_p

        return rhs_Cmat
    
    def _add_convective(self, Cmat, u, v, rhs_Cmat):
        flux = self.buffer
        ib = self.ib; ie = self.ie; jb = self.jb; je = self.je
        for l in range(6):
            # compute flux in x direction
            flux[jb:je + 1, ib - 1:ie + 1] = 0.5*u[jb:je + 1, ib - 1:ie + 1]*(Cmat[l, jb:je + 1, ib:ie + 2] + Cmat[l, jb:je + 1, ib - 1:ie + 1])
            flux[jb:je + 1, ib - 1:ie + 1] -= 0.5*np.abs(u[jb:je + 1, ib - 1:ie + 1])*(Cmat[l, jb:je + 1, ib:ie + 2] - Cmat[l, jb:je + 1, ib - 1:ie + 1])
            rhs_Cmat[l, jb:je + 1, ib:ie + 1] += (flux[jb:je + 1, ib - 1:ie] - flux[jb:je + 1, ib:ie + 1])/self.dx

            # compute flux in y direction
            flux[jb - 1:je + 1, ib:ie + 1] = 0.5*v[jb - 1:je + 1, ib:ie + 1]*(Cmat[l, jb:je + 2, ib:ie + 1] + Cmat[l, jb - 1:je + 1, ib:ie + 1])
            flux[jb - 1:je + 1, ib:ie + 1] -= 0.5*np.abs(v[jb - 1:je + 1, ib:ie + 1])*(Cmat[l, jb:je + 2, ib:ie + 1] - Cmat[l, jb - 1:je + 1, ib:ie + 1])
            rhs_Cmat[l, jb:je + 1, ib:ie + 1] += (flux[jb - 1:je, ib:ie + 1] - flux[jb:je + 1, ib:ie + 1])/self.dy

        return rhs_Cmat
    
    def _add_stretched(self, Cmat, dudx, dudy, dvdx, dvdy, rhs_Cmat):
        # no need to to interplate
        rhs_Cmat[0, :, :] += 2*Cmat[0, :, :]*dudx # C_11
        rhs_Cmat[1, :, :] += Cmat[1, :, :]*(dudx + dvdy) # C_12
        rhs_Cmat[2, :, :] += Cmat[2, :, :]*dudx # C_13
        rhs_Cmat[3, :, :] += 2*Cmat[3, :, :]*dvdy # C_22
        rhs_Cmat[4, :, :] += Cmat[4]*dvdy # C_23

        buffer = self.buffer


        # interpolate dudy from node to cell
        buffer = node2cell(dudy, buffer)
        # buffer = dudy
        rhs_Cmat[0, :, :] += 2*Cmat[1, :, :]*buffer
        rhs_Cmat[1, :, :] += Cmat[3, :, :]*buffer
        rhs_Cmat[2, :, :] += Cmat[4, :, :]*buffer

        # interpolate dvdx from node to cell
        buffer = node2cell(dvdx, buffer)
        # buffer = dvdx
        rhs_Cmat[1, :, :] += Cmat[0, :, :]*buffer
        rhs_Cmat[3, :, :] += 2*Cmat[1, :, :]*buffer
        rhs_Cmat[4, :, :] += Cmat[2, :, :]*buffer



        return rhs_Cmat
