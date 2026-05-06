def _checkinput(zi, Mi, z=False, verbose=None):
    """ Check and convert any input scalar or array to numpy array """
    # How many halo redshifts provided?
    zi = np.array(zi, ndmin=1, dtype=float)

    # How many halo masses provided?
    Mi = np.array(Mi, ndmin=1, dtype=float)

    # Check the input sizes for zi and Mi make sense, if not then exit unless
    # one axis is length one, then replicate values to the size of the other
    if (zi.size > 1) and (Mi.size > 1):
        if(zi.size != Mi.size):
            print("Error ambiguous request")
            print("Need individual redshifts for all haloes provided ")
            print("Or have all haloes at same redshift ")
            return(-1)
    elif (zi.size == 1) and (Mi.size > 1):
        if verbose:
            print("Assume zi is the same for all Mi halo masses provided")
        # Replicate redshift for all halo masses
        zi = np.ones_like(Mi)*zi[0]
    elif (Mi.size == 1) and (zi.size > 1):
        if verbose:
            print("Assume Mi halo masses are the same for all zi provided")
        # Replicate redshift for all halo masses
        Mi = np.ones_like(zi)*Mi[0]
    else:
        if verbose:
            print("A single Mi and zi provided")

    # Very simple test for size / type of incoming array
    # just in case numpy / list given
    if z is False:
        # Didn't pass anything, set zi = z
        lenzout = 1
    else:
        # If something was passed, convert to 1D NumPy array
        z = np.array(z, ndmin=1, dtype=float)
        lenzout = z.size

    return(zi, Mi, z, zi.size, Mi.size, lenzout)