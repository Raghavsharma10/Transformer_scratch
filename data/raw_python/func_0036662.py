def sinLdot_fc(tfdata, pfdata):
    """Apply sin of theta times the L operator to the data in the Fourier 
    domain."""
    
    dphi_fc(tfdata)
    
    sin_fc(pfdata)
    dtheta_fc(pfdata)
    
    return 1j * (tfdata - pfdata)