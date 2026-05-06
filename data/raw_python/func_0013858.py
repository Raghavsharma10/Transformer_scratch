def project_ecef_vector_onto_sc(inst, x_label, y_label, z_label, 
                                new_x_label, new_y_label, new_z_label,
                                meta=None):
    """Express input vector using s/c attitude directions
    
    x - ram pointing
    y - generally southward
    z - generally nadir
    
    Parameters
    ----------
    x_label : string
        Label used to get ECEF-X component of vector to be projected
    y_label : string
        Label used to get ECEF-Y component of vector to be projected
    z_label : string
        Label used to get ECEF-Z component of vector to be projected
    new_x_label : string
        Label used to set X component of projected vector
    new_y_label : string
        Label used to set Y component of projected vector
    new_z_label : string
        Label used to set Z component of projected vector
    meta : array_like of dicts (None)
        Dicts contain metadata to be assigned.
    """
    
    import pysatMagVect

    x, y, z = pysatMagVect.project_ecef_vector_onto_basis(inst[x_label], inst[y_label], inst[z_label],
                                                          inst['sc_xhat_ecef_x'], inst['sc_xhat_ecef_y'], inst['sc_xhat_ecef_z'],
                                                          inst['sc_yhat_ecef_x'], inst['sc_yhat_ecef_y'], inst['sc_yhat_ecef_z'],
                                                          inst['sc_zhat_ecef_x'], inst['sc_zhat_ecef_y'], inst['sc_zhat_ecef_z'])
    inst[new_x_label] = x
    inst[new_y_label] = y
    inst[new_z_label] = z
    
    if meta is not None:
        inst.meta[new_x_label] = meta[0]
        inst.meta[new_y_label] = meta[1]
        inst.meta[new_z_label] = meta[2]
    
    return