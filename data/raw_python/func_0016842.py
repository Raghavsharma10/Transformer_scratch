def parallactic_angles(times, antenna_positions, field_centre):
    """
    Computes parallactic angles per timestep for the given
    reference antenna position and field centre.

    Arguments:
        times: ndarray
            Array of unique times with shape (ntime,),
            obtained from TIME column of MS table
        antenna_positions: ndarray of shape (na, 3)
            Antenna positions, obtained from POSITION
            column of MS ANTENNA sub-table
        field_centre : ndarray of shape (2,)
            Field centre, should be obtained from MS PHASE_DIR

    Returns:
        An array of parallactic angles per time-step

    """
    import pyrap.quanta as pq

    try:
        # Create direction measure for the zenith
        zenith = pm.direction('AZEL','0deg','90deg')
    except AttributeError as e:
        if pm is None:
            raise ImportError("python-casacore import failed")

        raise

    # Create position measures for each antenna
    reference_positions = [pm.position('itrf',
        *(pq.quantity(x,'m') for x in pos))
        for pos in antenna_positions]

    # Compute field centre in radians
    fc_rad = pm.direction('J2000',
        *(pq.quantity(f,'rad') for f in field_centre))

    return np.asarray([
            # Set current time as the reference frame
            pm.do_frame(pm.epoch("UTC", pq.quantity(t, "s")))
            and
            [   # Set antenna position as the reference frame
                pm.do_frame(rp)
                and
                pm.posangle(fc_rad, zenith).get_value("rad")
                for rp in reference_positions
            ]
        for t in times])