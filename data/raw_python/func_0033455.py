def field_line_trace(init, date, direction, height, steps=None,
                     max_steps=1E4, step_size=10., recursive_loop_count=None, 
                     recurse=True):
    """Perform field line tracing using IGRF and scipy.integrate.odeint.
    
    Parameters
    ----------
    init : array-like of floats
        Position to begin field line tracing from in ECEF (x,y,z) km
    date : datetime or float
        Date to perform tracing on (year + day/365 + hours/24. + etc.)
        Accounts for leap year if datetime provided.
    direction : int
         1 : field aligned, generally south to north. 
        -1 : anti-field aligned, generally north to south.
    height : float
        Altitude to terminate trace, geodetic WGS84 (km)
    steps : array-like of ints or floats
        Number of steps along field line when field line trace positions should 
        be reported. By default, each step is reported; steps=np.arange(max_steps).
    max_steps : float
        Maximum number of steps along field line that should be taken
    step_size : float
        Distance in km for each large integration step. Multiple substeps
        are taken as determined by scipy.integrate.odeint
        
    Returns
    -------
    numpy array
        2D array. [0,:] has the x,y,z location for initial point
        [:,0] is the x positions over the integration.
        Positions are reported in ECEF (km).
        
    
    """
    
    if recursive_loop_count is None:  
        recursive_loop_count = 0
    #     
    if steps is None:
        steps = np.arange(max_steps)
    if not isinstance(date, float):
        # recast from datetime to float, as required by IGRF12 code
        doy = (date - datetime.datetime(date.year,1,1)).days
        # number of days in year, works for leap years
        num_doy_year = (datetime.datetime(date.year+1,1,1) - datetime.datetime(date.year,1,1)).days
        date = float(date.year) + float(doy)/float(num_doy_year) + float(date.hour + date.minute/60. + date.second/3600.)/24.
          
    trace_north = scipy.integrate.odeint(igrf.igrf_step, init.copy(),
                                         steps,
                                         args=(date, step_size, direction, height),
                                         full_output=False,
                                         printmessg=False,
                                         ixpr=False) #,
                                         # mxstep=500)
    
    # check that we reached final altitude
    check = trace_north[-1, :]
    x, y, z = ecef_to_geodetic(*check)        
    if height == 0:
        check_height = 1.
    else:
        check_height = height
    # fortran integration gets close to target height        
    if recurse & (z > check_height*1.000001):
        if (recursive_loop_count < 1000):
            # When we have not reached the reference height, call field_line_trace 
            # again by taking check value as init - recursive call
            recursive_loop_count = recursive_loop_count + 1
            trace_north1 = field_line_trace(check, date, direction, height,
                                            step_size=step_size, 
                                            max_steps=max_steps,
                                            recursive_loop_count=recursive_loop_count,
                                            steps=steps)
        else:
            raise RuntimeError("After 1000 iterations couldn't reach target altitude")
        return np.vstack((trace_north, trace_north1))
    else:
        # return results if we make it to the target altitude
        
        # filter points to terminate at point closest to target height
        # code below not correct, we want the first poiint that goes below target
        # height
        # code also introduces a variable length return, though I suppose
        # that already exists with the recursive functionality
        # x, y, z = ecef_to_geodetic(trace_north[:,0], trace_north[:,1], trace_north[:,2]) 
        # idx = np.argmin(np.abs(check_height - z)) 
        return trace_north