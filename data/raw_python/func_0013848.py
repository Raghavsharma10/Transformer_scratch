def load(fnames, tag=None, sat_id=None, obs_long=0., obs_lat=0., obs_alt=0., 
                                        TLE1=None, TLE2=None):
    """		          
    Returns data and metadata in the format required by pysat. Finds position		
    of satellite in both ECI and ECEF co-ordinates.
    
    Routine is directly called by pysat and not the user.		
    		
    Parameters		
    ----------		
    fnames : list-like collection		
        File name that contains date in its name. 		
    tag : string		
        Identifies a particular subset of satellite data		
    sat_id : string		
        Satellite ID			
    obs_long: float		
        Longitude of the observer on the Earth's surface		
    obs_lat: float		
        Latitude of the observer on the Earth's surface			
    obs_alt: float		
        Altitude of the observer on the Earth's surface		
    TLE1 : string
        First string for Two Line Element. Must be in TLE format	          
    TLE2 : string
        Second string for Two Line Element. Must be in TLE format	          
        
    Example
    -------
      inst = pysat.Instrument('pysat', 'sgp4', 
              TLE1='1 25544U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998',
              TLE2='2 25544  51.6402 181.0633 0004018  88.8954  22.2246 15.54059185113452')
      inst.load(2018, 1)
      
    """          
    
    import sgp4
    # wgs72 is the most commonly used gravity model in satellite tracking community
    from sgp4.earth_gravity import wgs72
    from sgp4.io import twoline2rv
    import ephem
    import pysatMagVect

    # TLEs (Two Line Elements for ISS)   
    # format of TLEs is fixed and available from wikipedia... 
    # lines encode list of orbital elements of an Earth-orbiting object 
    # for a given point in time        
    line1 = ('1 25544U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998')
    line2 = ('2 25544  51.6402 181.0633 0004018  88.8954  22.2246 15.54059185113452')
    # use ISS defaults if not provided by user
    if TLE1 is not None:
        line1 = TLE1
    if TLE2 is not None:
        line2 = TLE2
    
    # create satellite from TLEs and assuming a gravity model
    # according to module webpage, wgs72 is common
    satellite = twoline2rv(line1, line2, wgs72)

    # grab date from filename
    parts = os.path.split(fnames[0])[-1].split('-')
    yr = int(parts[0])
    month = int(parts[1])
    day = int(parts[2][0:2])
    date = pysat.datetime(yr, month, day)
    
    # create timing at 1 Hz (for 1 day)
    times = pds.date_range(start=date, end=date+pds.DateOffset(seconds=86399), freq='1S')
    # reduce requirements if on testing server
    # TODO Remove this when testing resources are higher
    on_travis = os.environ.get('ONTRAVIS') == 'True'
    if on_travis:
        times = times[0:100]
        
    # create list to hold satellite position, velocity
    position = []
    velocity = []
    for time in times:
        # orbit propagator - computes x,y,z position and velocity
        pos, vel = satellite.propagate(time.year, time.month, time.day, 
                                       time.hour, time.minute, time.second)
        # print (pos)
        position.extend(pos)
        velocity.extend(vel)
        
    # put data into DataFrame
    data = pysat.DataFrame({'position_eci_x': position[::3], 
                            'position_eci_y': position[1::3], 
                            'position_eci_z': position[2::3],
                            'velocity_eci_x': velocity[::3], 
                            'velocity_eci_y': velocity[1::3], 
                            'velocity_eci_z': velocity[2::3]}, 
                            index=times)
    data.index.name = 'Epoch'
    
    # add position and velocity in ECEF
    # add call for GEI/ECEF translation here
    # instead, since available, I'll use an orbit predictor from another
    # package that outputs in ECEF
    # it also supports ground station calculations
    
    # the observer's (ground station) position on the Earth surface
    site = ephem.Observer()
    site.lon = str(obs_long)   
    site.lat = str(obs_lat)   
    site.elevation = obs_alt 
    
    # The first parameter in readtle() is the satellite name
    sat = ephem.readtle('pysat' , line1, line2)
    output_params = []
    for time in times:
        lp = {}
        site.date = time
        sat.compute(site)
        # parameters relative to the ground station
        lp['obs_sat_az_angle'] = ephem.degrees(sat.az)
        lp['obs_sat_el_angle'] = ephem.degrees(sat.alt)
        # total distance away
        lp['obs_sat_slant_range'] = sat.range
        # satellite location 
        # sub latitude point
        lp['glat'] = np.degrees(sat.sublat)
        # sublongitude point
        lp['glong'] = np.degrees(sat.sublong)
        # elevation of sat in m, stored as km
        lp['alt'] = sat.elevation/1000.
        # get ECEF position of satellite
        lp['x'], lp['y'], lp['z'] = pysatMagVect.geodetic_to_ecef(lp['glat'], lp['glong'], lp['alt'])
        output_params.append(lp)
    output = pds.DataFrame(output_params, index=times)
    # modify input object to include calculated parameters
    data[['glong', 'glat', 'alt']] = output[['glong', 'glat', 'alt']]
    data[['position_ecef_x', 'position_ecef_y', 'position_ecef_z']] = output[['x', 'y', 'z']]
    data['obs_sat_az_angle'] = output['obs_sat_az_angle']
    data['obs_sat_el_angle'] = output['obs_sat_el_angle']
    data['obs_sat_slant_range'] = output['obs_sat_slant_range']
    return data, meta.copy()