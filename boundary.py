def set_pressure_bc(itype, p, ib, ie, jb, je):
    if itype == 0:
        p[jb:je + 1, ib - 1] = p[jb:je + 1, ie] # left (periodic)
        p[jb:je + 1, ie + 1] = p[jb:je + 1, ib] # right (periodic)
        p[jb - 1, :]         = p[jb, :]         # bottom (zero gradient)
        p[je + 1, :]         = p[je, :]         # top (zero gradient)
    elif itype == 1:
        p[jb:je + 1, ib - 1] = p[jb:je + 1, ib] # left (zero gradient)
        p[jb:je + 1, ie + 1] = p[jb:je + 1, ie] # right (zero gradient)
        p[jb - 1, :]         = p[jb, :]         # bottom (zero gradient)
        p[je + 1, :]         = p[je, :]         # top (zero gradient)
    
    return p

def set_Cmat_bc(itype, Cmat, ib, ie, jb, je):
    if itype == 0:
        Cmat[:, jb:je + 1, ib - 1] = Cmat[:, jb:je + 1, ie] # left (periodic)
        Cmat[:, jb:je + 1, ie + 1] = Cmat[:, jb:je + 1, ib] # right (periodic)
        Cmat[:, jb - 1, :]         = Cmat[:, jb, :]         # bottom (zero gradient)
        Cmat[:, je + 1, :]         = Cmat[:, je, :]         # top (zero gradient)
    elif itype == 1:
        Cmat[:, jb:je + 1, ib - 1] = Cmat[:, jb:je + 1, ib] # left (zero gradient)
        Cmat[:, jb:je + 1, ie + 1] = Cmat[:, jb:je + 1, ie] # right (zero gradient)
        Cmat[:, jb - 1, :]         = Cmat[:, jb, :]         # bottom (zero gradient)
        Cmat[:, je + 1, :]         = Cmat[:, je, :]         # top (zero gradient)

    return Cmat

def set_velocity_bc(itype, u, v,
                    uib, uie, ujb, uje,
                    vib, vie, vjb, vje,
                    Utop=None):
    if itype == 0:
        u[ujb:uje + 1, uib - 1] = u[ujb:uje + 1, uie] # left (periodic)
        u[ujb:uje + 1, uie + 1] = u[ujb:uje + 1, uib] # right (periodic)
        u[ujb - 1, :]           = -u[ujb, :]          # bottom (no-slip wall)
        u[uje + 1, :]           = -u[uje, :]          # top (no-slip wall)

        v[vjb:vje + 1, vib - 1] = v[vjb:vje + 1, vie] # left (periodic)
        v[vjb:vje + 1, vie + 1] = v[vjb:vje + 1, vib] # right (periodic)
        v[vjb - 1, :]           = 0.0                 # bottom (no-slip wall)
        v[vje + 1, :]           = 0.0                 # top (no-slip wall)

    elif itype == 1:
        u[ujb:uje + 1, uib - 1] = 0.0                 # left (no-slip wall)
        u[ujb:uje + 1, uie + 1] = 0.0                 # right (no-slip wall)
        u[ujb - 1, :]           = -u[ujb, :]          # bottom (no-slip wall)
        u[uje + 1, :]           = 2*Utop - u[uje, :]  # top (no-slip wall)

        v[vjb:vje + 1, vib - 1] = -v[vjb:vje + 1, vib] # left (no-slip wall)
        v[vjb:vje + 1, vie + 1] = -v[vjb:vje + 1, vie] # right (no-slip wall)
        v[vjb - 1, :]           = 0.0                  # bottom (no-slip wall)
        v[vje + 1, :]           = 0.0                  # top (no-slip wall)

    
    return u, v
