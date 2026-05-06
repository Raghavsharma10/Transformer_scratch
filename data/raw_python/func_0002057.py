def calc_offset(cube):
    """Calculate an offset.

    Calculate offset from the side of data so that at least 200 image pixels are in the MAD stats.

    Parameters
    ==========
    cube : pyciss.ringcube.RingCube
        Cubefile with ring image
    """
    i = 0
    while pd.Series(cube.img[:, i]).count() < 200:
        i += 1
    return max(i, 20)