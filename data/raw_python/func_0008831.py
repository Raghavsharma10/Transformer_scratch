def save_as_image(region, filename):
    """
    Convert a MIMAS region (.mim) file into a image (eg .png)

    Parameters
    ----------
    region : :class:`AegeanTools.regions.Region`
        Region of interest.

    filename : str
        Output filename.
    """
    import healpy as hp
    pixels = list(region.get_demoted())
    order = region.maxdepth
    m = np.arange(hp.nside2npix(2**order))
    m[:] = 0
    m[pixels] = 1
    hp.write_map(filename, m, nest=True, coord='C')
    return