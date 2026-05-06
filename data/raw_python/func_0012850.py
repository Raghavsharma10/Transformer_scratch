def unit_normal(apnt, bpnt, cpnt):
    """unit normal"""
    xvar = np.tinylinalg.det([
        [1, apnt[1], apnt[2]], [1, bpnt[1], bpnt[2]], [1, cpnt[1], cpnt[2]]])
    yvar = np.tinylinalg.det([
        [apnt[0], 1, apnt[2]], [bpnt[0], 1, bpnt[2]], [cpnt[0], 1, cpnt[2]]])
    zvar = np.tinylinalg.det([
        [apnt[0], apnt[1], 1], [bpnt[0], bpnt[1], 1], [cpnt[0], cpnt[1], 1]])
    magnitude = (xvar**2 + yvar**2 + zvar**2)**.5
    if magnitude < 0.00000001:
        mag = (0, 0, 0)
    else: mag = (xvar/magnitude, yvar/magnitude, zvar/magnitude)
    return mag