from boundary import set_velocity_bc, set_pressure_bc, set_Cmat_bc
from poisson_solver import PoissonSolver
import numpy as np
from utils import calculate_velocity_gradient, calculate_node_velocity, calculate_cell_velocity
from readwrite import print_step
from initial import initialize_poiseuille
from rheology import Rheology
from navier_stokes import NS

import os
if __name__ == "__main__":
    '''
    Case specific input paramters
    1. itype  :
           0 = start-up planar Poiseuille flow (driven by -dP/dx = 0.02 Pa/m)
           1 = cavity flow (driven by lid moving at U = 1 m/s)
    '''
    itype = 0
    is_forced = True
    force = 0.02  # None for cavity
    Utop  = None  # 1    for cavity


    '''
    Dimension and geometry
    '''
    Nx = 16; Ny = 128
    Lx = 0.25; Ly = 2.0
    dx = Lx/Nx; dy = Ly/Ny

    '''
    Fluid
    ifluid:
      0 = Newtonian with viscosity mu_s
      1 = Oldroyd-B
    '''
    ifluid = 0
    mu_s  = 0.01
    mu_p  = 0.0025 # polymer viscosity (0 for Newtonian)
    tau_p = 10.0   # polymer relaxation time

    '''
    time control
    '''
    dt = 0.01
    t_final = 1000.0
    nt = int(t_final/dt)

    '''
    IO stuff
    '''
    output = "planar_Poiseuille_Newtonian"
    if not os.path.exists(output):
        os.mkdir(output)
    iprint = 500

    '''
    Index system (all end index is inclusive)
    In Python: a[jb:je + 1, ib:ie + 1] = 1
    In Fortran:
    do j = jb, je
      do i = ib, ie
        a(i, j) = 1
      end do
    end do
    '''
    # all field will have shape (Ntot_y, Ntot_x) including 1 layer of ghost cells
    Ntot_x = Nx + 2; Ntot_y = Ny + 2
    # begin and end index for pressure and conformation tensor
    ib = 1; ie = Nx; jb = 1; je = Ny
    # begin and end index for u
    uib = 1
    if itype == 0:
        uie = Nx # due to periodic
    elif itype == 1:
        uie = Nx - 1
    ujb = 1; uje = Ny
    # begin and end index for v
    vib = 1; vie = Nx; vjb = 1; vje = Ny - 1

    '''
    Memory for the main program
    '''
    # velocity field
    u = np.zeros((Ntot_y, Ntot_x)); rhs_u  = np.zeros((Ntot_y, Ntot_x))
    v = np.zeros((Ntot_y, Ntot_x)); rhs_v  = np.zeros((Ntot_y, Ntot_x))

    # velocity at node center
    u_n = np.zeros((Ntot_y, Ntot_x)); v_n = np.zeros((Ntot_y, Ntot_x))
    # velocity at cell center
    u_c = np.zeros((Ntot_y, Ntot_x)); v_c = np.zeros((Ntot_y, Ntot_x))

    # pressure Poisson equation
    p = np.zeros((Ntot_y, Ntot_x)); p_corr = np.zeros((Ntot_y, Ntot_x))
    div = np.zeros((Ntot_y, Ntot_x))

    # velocity gradient and stress tensor
    sigma_xx = np.zeros((Ntot_y, Ntot_x)) # cell center
    sigma_xy = np.zeros((Ntot_y, Ntot_x)) # node center
    sigma_yy = np.zeros((Ntot_y, Ntot_x)) # cell center
    dudx = np.zeros((Ntot_y, Ntot_x)) # cell center
    dudy = np.zeros((Ntot_y, Ntot_x)) # node center
    dvdx = np.zeros((Ntot_y, Ntot_x)) # node center
    dvdy = np.zeros((Ntot_y, Ntot_x)) # cell center

    # conformation tensor
    # C_11, C_12, C_13, C_22, C_23, C_33
    #  0     1     2     3     4     5
    Cmat    = np.zeros((6, Ntot_y, Ntot_x)) # cell center
    rhs_Cmat = np.zeros((6, Ntot_y, Ntot_x))

    '''
    select Pressure solver
    '''
    if itype == 0:
        p_solver = PoissonSolver(Nx, Ny, dx, dy, BC="P-N")
    elif itype == 1:
        p_solver = PoissonSolver(Nx, Ny, dx, dy, BC="N-N")

    '''
    setup momentum solver
    '''
    mom_solver = NS(dx, dy, Ntot_x, Ntot_y, uib, uie, ujb, uje, vib, vie, vjb, vje)


    '''
    setup the Rheology model
    '''
    fluid = Rheology(Ntot_x, Ntot_y)

    '''
    Initialization
    '''
    if itype == 0:
        u, v, p, Cmat = initialize_poiseuille(u, v, p, Cmat)


    '''
    Get ready for time loop (also called after each iteration)
    '''
    u, v = set_velocity_bc(itype, u, v, uib, uie, ujb, uje, vib, vie, vjb, vje)
    p = set_pressure_bc(itype, p, ib, ie, jb, je)
    Cmat = set_Cmat_bc(itype, Cmat, ib, ie, jb, je)
    u_n, v_n = calculate_node_velocity(u, v, u_n, v_n)
    u_c, v_c = calculate_cell_velocity(u, v, u_c, v_c)
    dudx, dudy, dvdx, dvdy = calculate_velocity_gradient(u, v, dx, dy, dudx, dudy, dvdx, dvdy)
    div = dudx + dvdy

    # calculate the stress
    if ifluid == 0:
        sigma_xx, sigma_xy, sigma_yy = fluid.calculate_newtonian_stress(mu_s,
                                                                        dudx, dudy, dvdx, dvdy,
                                                                        sigma_xx, sigma_xy, sigma_yy)
    elif ifluid == 1:
        sigma_xx, sigma_xy, sigma_yy = fluid.calculate_Oldroyd_B_stress(mu_s, mu_p, tau_p,
                                                                        Cmat, dudx, dudy, dvdx, dvdy,
                                                                        sigma_xx, sigma_xy, sigma_yy)
        
    tnow   = 0.0
    for it in range(nt):
        if it % iprint == 0:
            print_step(it, tnow, u[1:-1, :-1], v[:-1, 1:-1], div[1:-1, 1:-1])

        
        '''
        Predictor
        '''
        rhs_u, rhs_v = mom_solver.calculate_RHS_NS(u_n, v_n, u_c, v_c, p,
                                                   sigma_xx, sigma_xy, sigma_yy,
                                                   rhs_u, rhs_v)
        # add body force
        if is_forced:
            rhs_u += force

        # now u and v is the tentative velocity
        u += dt*rhs_u
        v += dt*rhs_v
        u, v = set_velocity_bc(itype, u, v, uib, uie, ujb, uje, vib, vie, vjb, vje)

        '''
        Poisson equation
        '''
        dudx, dudy, dvdx, dvdy = calculate_velocity_gradient(u, v, dx, dy, dudx, dudy, dvdx, dvdy)
        div = (dudx + dvdy)/dt # RHS
        p_corr[jb:je + 1, ib:ie + 1] = p_solver.solve(div[jb:je + 1, ib:ie + 1])
        set_pressure_bc(itype, p, ib, ie, jb, je)

        '''
        corrector step
        '''
        u[ujb:uje + 1, uib:uie + 1] += dt*(p_corr[ujb:uje + 1, uib:uie + 1] - p_corr[ujb:uje + 1, uib + 1:uie + 2])/dx
        v[vjb:vje + 1, vib:vie + 1] += dt*(p_corr[vjb:vje + 1, vib:vie + 1] - p_corr[vjb + 1:vje + 2, vib:vie + 1])/dy

        tnow += dt

        '''
        Prepare for the next loop
        '''
        u, v = set_velocity_bc(itype, u, v, uib, uie, ujb, uje, vib, vie, vjb, vje)
        p = set_pressure_bc(itype, p, ib, ie, jb, je)
        Cmat = set_Cmat_bc(itype, Cmat, ib, ie, jb, je)
        u_n, v_n = calculate_node_velocity(u, v, u_n, v_n)
        u_c, v_c = calculate_cell_velocity(u, v, u_c, v_c)
        dudx, dudy, dvdx, dvdy = calculate_velocity_gradient(u, v, dx, dy, dudx, dudy, dvdx, dvdy)
        div = dudx + dvdy
        # calculate the stress
        if ifluid == 0:
            sigma_xx, sigma_xy, sigma_yy = fluid.calculate_newtonian_stress(mu_s,
                                                                            dudx, dudy, dvdx, dvdy,
                                                                            sigma_xx, sigma_xy, sigma_yy)
        elif ifluid == 1:
            sigma_xx, sigma_xy, sigma_yy = fluid.calculate_Oldroyd_B_stress(mu_s, mu_p, tau_p,
                                                                           Cmat, dudx, dudy, dvdx, dvdy,
                                                                           sigma_xx, sigma_xy, sigma_yy)

    it += 1
    print_step(it, tnow, u[1:-1, :-1], v[:-1, 1:-1], div[1:-1, 1:-1])




    
    
        


