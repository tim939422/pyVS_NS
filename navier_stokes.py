import numpy as np


class NS:
    def __init__(self, dx, dy,
                 Ntot_x, Ntot_y,
                 uib, uie, ujb, uje,
                 vib, vie, vjb, vje):
        self.dx = dx
        self.dy = dy

        self.uib = uib; self.uie = uie; self.ujb = ujb; self.uje = uje
        self.vib = vib; self.vie = vie; self.vjb = vjb; self.vje = vje

        # buffer for numerical flux of convective term
        self.flux = np.zeros((Ntot_y, Ntot_x))

    def calculate_RHS_NS(self, u_n, v_n, u_c, v_c,
                         p,
                         sigma_xx, sigma_xy, sigma_yy,
                         rhs_u, rhs_v):
        rhs_u, rhs_v = self._add_convective(u_n, v_n, u_c, v_c, rhs_u, rhs_v)
        rhs_u, rhs_v = self._add_pgrad(p, rhs_u, rhs_v)
        rhs_u, rhs_v = self._add_viscous(sigma_xx, sigma_xy, sigma_yy, rhs_u, rhs_v)

        return rhs_u, rhs_v
    
    # to mimic zero set, this must be called first
    def _add_convective(self, u_n, v_n, u_c, v_c,
                        rhs_u, rhs_v):
        uib = self.uib; uie = self.uie; ujb = self.ujb; uje = self.uje
        vib = self.vib; vie = self.vie; vjb = self.vjb; vje = self.vje

        # x momentum -d(u^2)/dx
        self.flux = u_c**2
        rhs_u[ujb:uje + 1, uib:uie + 1] = (self.flux[ujb:uje + 1, uib:uie + 1] - self.flux[ujb:uje + 1, uib + 1:uie + 2])/self.dx
        
        self.flux = u_n*v_n
        # x momentum - d(uv)/dy
        rhs_u[ujb:uje + 1, uib:uie + 1] += (self.flux[ujb - 1:uje, uib:uie + 1] - self.flux[ujb:uje + 1, uib:uie + 1])/self.dy

        # y momentum -d(vu)/dx
        rhs_v[vjb:vje + 1, vib:vie + 1] = (self.flux[vjb:vje + 1, vib - 1:vie] - self.flux[vjb:vje + 1, vib:vie + 1])/self.dx

        self.flux = v_c**2
        # y momentum -d(vu)/dx - d(v^2)/dy
        rhs_v[vjb:vje + 1, vib:vie + 1] += (self.flux[vjb:vje + 1, vib:vie + 1] - self.flux[vjb + 1:vje + 2, vib:vie + 1])/self.dy

        return rhs_u, rhs_v

    def _add_viscous(self, sigma_xx, sigma_xy, sigma_yy,
                     rhs_u, rhs_v):
        uib = self.uib; uie = self.uie; ujb = self.ujb; uje = self.uje
        vib = self.vib; vie = self.vie; vjb = self.vjb; vje = self.vje

        # d(sigma_xx)/dx
        rhs_u[ujb:uje + 1, uib:uie + 1] += (sigma_xx[ujb:uje + 1, uib + 1:uie + 2] - sigma_xx[ujb:uje + 1, uib:uie + 1])/self.dx
        # d(sigma_xy)/dy
        rhs_u[ujb:uje + 1, uib:uie + 1] += (sigma_xy[ujb:uje + 1, uib:uie + 1] - sigma_xy[ujb - 1:uje, uib:uie + 1])/self.dy

        # d(sigma_yx)/dx = d(sigma_xy)/dx                  (i, j)                             (i-1, j)
        rhs_v[vjb:vje + 1, vib:vie + 1] += (sigma_xy[vjb:vje + 1, vib:vie + 1] - sigma_xy[vjb:vje + 1, vib - 1:vie])/self.dx
        # d(sigma_yy)/dy       (i, j)                   (i,j+1)                                    (i, j)
        rhs_v[vjb:vje + 1, vib:vie + 1] += (sigma_yy[vjb + 1:vje + 2, vib:vie + 1] - sigma_yy[vjb:vje + 1, vib:vie + 1])/self.dy

        return rhs_u, rhs_v
        

    def _add_pgrad(self, p, rhs_u, rhs_v):
        uib = self.uib; uie = self.uie; ujb = self.ujb; uje = self.uje
        vib = self.vib; vie = self.vie; vjb = self.vjb; vje = self.vje

        # -dp/dx      (i, j)                       (i, j)                          (i + 1, j)
        rhs_u[ujb:uje + 1, uib:uie + 1] += (p[ujb:uje + 1, uib:uie + 1] - p[ujb:uje + 1, uib + 1:uie + 2])/self.dx
        # -dp/dy      (i, j)                       (i, j)                          (i, j + 1)
        rhs_v[vjb:vje + 1, vib:vie + 1] += (p[vjb:vje + 1, vib:vie + 1] - p[vjb + 1:vje + 2, vib:vie + 1])/self.dy

        return rhs_u, rhs_v