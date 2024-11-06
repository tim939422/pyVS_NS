def print_step(it, tnow, u, v, div):
    print("")
    print(f"it = {it:5d}, t = {tnow:.3e}")
    print("")
    print(f"div : {div.min():.5e} (min) {div.max():.5e} (max)")
    print(f"u   : {u.min():.5e} (min) {u.max():.5e} (max)")
    print(f"v   : {v.min():.5e} (min) {v.max():.5e} (max)")
