def calculate_velocity_gradient(u, v, dx, dy, dudx, dudy, dvdx, dvdy):
    # velocity gradient at all available cell/node
    # 2nd order central difference

    # cell center
    dudx[:, 1:] = (u[:, 1:] - u[:, :-1])/dx # i = 1 -> Nx + 1, j = 0 -> Ny + 1
    dvdy[1:, :] = (v[1:, :] - v[:-1, :])/dy # i = 0 -> Nx + 1, j = 1 -> Ny + 1 

    # node center
    dudy[:-1, :] = (u[1:, :] - u[:-1, :])/dy # i = 0 -> Nx + 1, j = 0 -> Ny
    dvdx[:, :-1] = (v[:, 1:] - v[:, :-1])/dx # i = 0 -> Nx, j = 0 -> Ny + 1

    return dudx, dudy, dvdx, dvdy

'''
The node and cell velocity is needed for convective terms in NS equation.
Currently, we are only working with simple average, i.e., central scheme.
This will cause problem at high Re.
'''
def calculate_node_velocity(u, v, u_n, v_n):
    u_n[:-1, :-1] = 0.5*(u[:-1,:-1] + u[1:,:-1])
    v_n[:-1, :-1] = 0.5*(v[:-1, :-1] + v[:-1, 1:])
    return u_n, v_n

def calculate_cell_velocity(u, v, u_c, v_c):
    u_c[:, 1:] = 0.5*(u[:, :-1] + u[:, 1:])
    v_c[1:, :] = 0.5*(v[:-1, :] + v[1:, :])

    return u_c, v_c

'''
Interpolate cell center data (for example, p or Cmat) to node
or backwards
'''
def cell2node(phi_c, phi_n):
    phi_n[:-1, :-1] = 0.25*(phi_c[:-1, :-1] + phi_c[1:, :-1] + phi_c[:-1, 1:] + phi_c[1:, 1:])
    return phi_n

def node2cell(phi_n, phi_c):
    phi_c[1:-1, 1:-1] = 0.25*(phi_n[1:-1, 1:-1] + phi_n[:-2, 1:-1] + phi_n[:-2, :-2] + phi_n[1:-1, :-2])
    return phi_n




# Function to parse the configuration file
def read_config(filename):
    config = {}
    with open(filename, 'r') as f:
        for line in f:
            # Skip comments and blank lines
            if line.startswith("#") or line.strip() == "":
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            # Convert values to appropriate types
            try:
                config[key] = eval(value)  # Use eval to parse numbers and booleans
            except NameError:
                config[key] = value  # For string values like "output"
    return config