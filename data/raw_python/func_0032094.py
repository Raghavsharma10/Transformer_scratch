def parse_file(infile, exit_on_error=True):
    """Parse a comma-separated file with columns "ra,dec,magnitude".
    """
    try:
        a, b, mag = np.atleast_2d(
                            np.genfromtxt(
                                        infile,
                                        usecols=[0, 1, 2],
                                        delimiter=','
                                        )
                    ).T
    except IOError as e:
        if exit_on_error:
            logger.error("There seems to be a problem with the input file, "
                         "the format should be: RA_degrees (J2000), Dec_degrees (J2000), "
                         "Magnitude. There should be no header, columns should be "
                         "separated by a comma")
            sys.exit(1)
        else:
            raise e
    return a, b, mag