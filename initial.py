def initialize_poiseuille(u, v, p, Cmat):
    
    u[:, :] = 0.0
    v[:, :] = 0.0
    p[:, :] = 0.0

    # conformation tensor
    # C_11, C_12, C_13, C_22, C_23, C_33
    #  0     1     2     3     4     5
    # C = I at rest
    Cmat[:, :, :] = 0.0
    Cmat[0, :, :] = 1.0
    Cmat[3, :, :] = 1.0
    Cmat[5, :, :] = 1.0
    
    return u, v, p, Cmat

def initialize_cavity(u, v, p, Cmat):
    u[:, :] = 0.0
    v[:, :] = 0.0
    p[:, :] = 0.0

    # conformation tensor
    # C_11, C_12, C_13, C_22, C_23, C_33
    #  0     1     2     3     4     5
    # C = I at rest
    Cmat[:, :, :] = 0.0
    Cmat[0, :, :] = 1.0
    Cmat[3, :, :] = 1.0
    Cmat[5, :, :] = 1.0
    
    return u, v, p, Cmat