def calculate_dayside_reconnection(inst):
    """ Calculate the dayside reconnection rate (Milan et al. 2014)

    Parameters
    -----------
    inst : pysat.Instrument
        Instrument with OMNI HRO data, requires BYZ_GSM and clock_angle

    Notes
    --------
    recon_day = 3.8 Re (Vx / 4e5 m/s)^1/3 Vx B_yz (sin(theta/2))^9/2
    """
    rearth = 6371008.8
    sin_htheta = np.power(np.sin(np.radians(0.5 * inst['clock_angle'])), 4.5)
    byz = inst['BYZ_GSM'] * 1.0e-9
    vx = inst['flow_speed'] * 1000.0
    
    recon_day = 3.8 * rearth * vx * byz * sin_htheta * np.power((vx / 4.0e5),
                                                                1.0/3.0)
    inst['recon_day'] = pds.Series(recon_day, index=inst.data.index)
    return