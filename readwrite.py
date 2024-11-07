import h5py
import numpy as np

def print_step(it, tnow, u, v, div):
    print("")
    print(f"it = {it:5d}, t = {tnow:.3e}")
    print("")
    print(f"div : {div.min():.5e} (min) {div.max():.5e} (max)")
    print(f"u   : {u.min():.5e} (min) {u.max():.5e} (max)")
    print(f"v   : {v.min():.5e} (min) {v.max():.5e} (max)")

def write_string_attr(id, attr, string):
    s = ()
    tid = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
    tid.set_strpad(h5py.h5t.STR_NULLTERM)
    space = h5py.h5s.create_simple(s)


    buffer = np.frombuffer(string.encode("ascii"), dtype="S" + str(len(string)))
    tid.set_size(len(string) + 1)
    aid = h5py.h5a.create(id, attr.encode("ascii"), tid, space) 
    aid.write(buffer)

def write_visu(fname, it, tnow, Nx, Ny, Lx, Ly, u_n, v_n, p_c, Cmat=None):
    with h5py.File(fname, "w") as f:
        '''Create mesh
        '''
        g = f.create_group("unigrid")

        # write string
        write_string_attr(g.id, "vsType", "mesh")
        write_string_attr(g.id, "vsKind", "uniform")
        write_string_attr(g.id, "vsIndexOrder", "compMinorF")

        # write integer arrays
        g.attrs["GridPoints"] = np.int32([Nx + 1, Ny + 1])
        g.attrs["vsNumCells"] = np.int32([Nx, Ny])
        g.attrs["vsStartCell"] = np.int32([0, 0])

        # write float array
        g.attrs["vsLowerBounds"] = np.float64([0.0, 0.0])
        g.attrs["vsUpperBounds"] = np.float64([Lx, Ly])

        '''
        write time information
        '''
        g = g.create_group("time")
        write_string_attr(g.id, "vsType", "time")
        g.attrs["vsTime"] = np.float64([tnow])
        g.attrs["vsStep"] = np.int32([it])

        '''
        zonal (cell center) data for pressure
        '''
        d = f.create_dataset("p", data=p_c)
        write_string_attr(d.id, "vsType", "variable")
        write_string_attr(d.id, "vsMesh", "/unigrid")
        write_string_attr(d.id, "vsIndexOrder", "compMinorF")
        write_string_attr(d.id, "vsCentering", "zonal")


        '''
        velocity vector
        '''
        g = f.create_group("velocity")

        gv = g.create_group("velocity")
        write_string_attr(gv.id, "vsType", "vsVars")
        write_string_attr(gv.id, "velocity", "{<velocity/u>,<velocity/v>}")

        d = g.create_dataset("u", data=u_n)
        write_string_attr(d.id, "vsType", "variable")
        write_string_attr(d.id, "vsMesh", "/unigrid")
        write_string_attr(d.id, "vsIndexOrder", "compMinorF")


        d = g.create_dataset("v", data=v_n)
        write_string_attr(d.id, "vsType", "variable")
        write_string_attr(d.id, "vsMesh", "/unigrid")
        write_string_attr(d.id, "vsIndexOrder", "compMinorF")

        if Cmat is not None:
            names = ["C11", "C12", "C13", "C22", "C23", "C33"]
            for l in range(6):
                d = f.create_dataset(names[l], data=Cmat[l, :, :])
                write_string_attr(d.id, "vsType", "variable")
                write_string_attr(d.id, "vsMesh", "/unigrid")
                write_string_attr(d.id, "vsIndexOrder", "compMinorF")
                write_string_attr(d.id, "vsCentering", "zonal")