from boundary import set_velocity_bc, set_pressure_bc, set_Cmat_bc
from poisson_solver import PoissonSolver
import numpy as np
from utils import calculate_velocity_gradient, calculate_node_velocity, calculate_cell_velocity, cell2node
from utils import read_config
from readwrite import print_step, write_visu
from initial import initialize_poiseuille, initialize_cavity
from rheology import Rheology
from navier_stokes import NS
from conformation import CTensor
import h5py
import os
import sys
if __name__ == "__main__":
    fname = sys.argv[1]
    # Load parameters from the config file
    params = read_config(fname)
    # Assign variables from config
    itype = params.get("itype")
    is_forced = params.get("is_forced")
    force = params.get("force")
    Utop = params.get("Utop")

    Nx = params.get("Nx")
    Ny = params.get("Ny")
    Lx = params.get("Lx")
    Ly = params.get("Ly")
    dx = Lx / Nx
    dy = Ly / Ny

    ifluid = params.get("ifluid")
    mu_s = params.get("mu_s")
    mu_p = params.get("mu_p")
    tau_p = params.get("tau_p")

    dt = params.get("dt")
    t_final = params.get("t_final")
    nt = int(t_final / dt)

    output = params.get("output")
    if not os.path.exists(output):
        os.mkdir(output)
    iprint = params.get("iprint")
    ivisu = params.get("ivisu")

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
    p_n = np.zeros((Ntot_y, Ntot_x))

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
    setup conformation tensor solver
    '''
    ctensor_solver = CTensor(tau_p, dx, dy, Ntot_x, Ntot_y, ib, ie, jb, je)

    '''
    setup the Rheology model
    '''
    fluid = Rheology(Ntot_x, Ntot_y)

    '''
    Initialization
    '''
    if itype == 0:
        u, v, p, Cmat = initialize_poiseuille(u, v, p, Cmat)
    elif itype == 1:
        u, v, p, Cmat = initialize_cavity(u, v, p, Cmat)

    '''
    Get ready for time loop (also called after each iteration)
    '''
    u, v = set_velocity_bc(itype, u, v, uib, uie, ujb, uje, vib, vie, vjb, vje, Utop)
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
        if it % ivisu == 0:
            fname = os.path.join(output, f"visu_{it:06d}.h5")
            write_visu(fname, it, tnow, Nx, Ny, Lx, Ly, u_n[:-1, :-1], v_n[:-1, :-1], p[1:-1, 1:-1], Cmat[:, 1:-1, 1:-1])
            print(f"Saving {fname}")
        
        '''
        advance C
        '''
        if ifluid == 1:
            rhs_Cmat = ctensor_solver.calculate_RHS(Cmat, u, v, dudx, dudy, dvdx, dvdy, rhs_Cmat)
            Cmat += dt*rhs_Cmat
            Cmat = set_Cmat_bc(itype, Cmat, ib, ie, jb, je)

        
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
        u, v = set_velocity_bc(itype, u, v, uib, uie, ujb, uje, vib, vie, vjb, vje, Utop)

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
        p[jb:je + 1, ib:ie + 1] += p_corr[jb:je + 1, ib:ie + 1] - p_corr[jb, ib]

        tnow += dt

        '''
        Prepare for the next loop
        '''
        u, v = set_velocity_bc(itype, u, v, uib, uie, ujb, uje, vib, vie, vjb, vje, Utop)
        p = set_pressure_bc(itype, p, ib, ie, jb, je)
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


    fname = os.path.join(output, "restart.h5")
    with h5py.File(fname, "w") as f:
        f.create_dataset("u", data=u[1:-1, :-1])
        f.create_dataset("v", data=v[:-1, 1:-1])
        f.create_dataset("p", data=p[1:-1, 1:-1])

    fname = os.path.join(output, f"visu_{it:06d}.h5")
    write_visu(fname, it, tnow, Nx, Ny, Lx, Ly, u_n[:-1, :-1], v_n[:-1, :-1], p[1:-1, 1:-1], Cmat[:, 1:-1, 1:-1])

    


    
    
        


