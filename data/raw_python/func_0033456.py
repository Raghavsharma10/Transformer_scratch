def full_field_line(init, date, height, step_size=100., max_steps=1000, 
                    steps=None, **kwargs):
    """Perform field line tracing using IGRF and scipy.integrate.odeint.
    
    Parameters
    ----------
    init : array-like of floats
        Position to begin field line tracing from in ECEF (x,y,z) km
    date : datetime or float
        Date to perform tracing on (year + day/365 + hours/24. + etc.)
        Accounts for leap year if datetime provided.
    height : float
        Altitude to terminate trace, geodetic WGS84 (km)
    max_steps : float
        Maximum number of steps along field line that should be taken
    step_size : float
        Distance in km for each large integration step. Multiple substeps
        are taken as determined by scipy.integrate.odeint
    steps : array-like of ints or floats
        Number of steps along field line when field line trace positions should 
        be reported. By default, each step is reported; steps=np.arange(max_steps).
        Two traces are made, one north, the other south, thus the output array
        could have double max_steps, or more via recursion.
        
    Returns
    -------
    numpy array
        2D array. [0,:] has the x,y,z location for southern footpoint
        [:,0] is the x positions over the integration.
        Positions are reported in ECEF (km).
        
    
    """
    
    if steps is None:
        steps = np.arange(max_steps)
    # trace north, then south, and combine
    trace_south = field_line_trace(init, date, -1., height, 
                                   steps=steps,
                                   step_size=step_size, 
                                   max_steps=max_steps, 
                                   **kwargs)
    trace_north = field_line_trace(init, date, 1., height, 
                                   steps=steps,
                                   step_size=step_size, 
                                   max_steps=max_steps, 
                                   **kwargs)
    # order of field points is generally along the field line, south to north
    # don't want to include the initial point twice
    trace = np.vstack((trace_south[::-1][:-1,:], trace_north))
    return trace