def mim2reg(mimfile, regfile):
    """
    Convert a MIMAS region (.mim) file into a DS9 region (.reg) file.

    Parameters
    ----------
    mimfile : str
        Input file in MIMAS format.

    regfile : str
        Output file.

    """
    region = Region.load(mimfile)
    region.write_reg(regfile)
    logging.info("Converted {0} -> {1}".format(mimfile, regfile))
    return