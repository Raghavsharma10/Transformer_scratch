def calculate_clock_angle(inst):
    """ Calculate IMF clock angle and magnitude of IMF in GSM Y-Z plane

    Parameters
    -----------
    inst : pysat.Instrument
        Instrument with OMNI HRO data
    """
    
    # Calculate clock angle in degrees
    clock_angle = np.degrees(np.arctan2(inst['BY_GSM'], inst['BZ_GSM']))
    clock_angle[clock_angle < 0.0] += 360.0
    inst['clock_angle'] = pds.Series(clock_angle, index=inst.data.index)

    # Calculate magnitude of IMF in Y-Z plane
    inst['BYZ_GSM'] = pds.Series(np.sqrt(inst['BY_GSM']**2 +
                                         inst['BZ_GSM']**2),
                                       index=inst.data.index)

    return