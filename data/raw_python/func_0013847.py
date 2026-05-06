def init(self):
    """
    Adds custom calculations to orbit simulation.
    This routine is run once, and only once, upon instantiation.
    
    Adds quasi-dipole coordiantes, velocity calculation in ECEF coords,
    adds the attitude vectors of spacecraft assuming x is ram pointing and
    z is generally nadir, adds ionospheric parameters from the Interational
    Reference Ionosphere (IRI), as well as simulated winds from the
    Horiontal Wind Model (HWM).
    
    """
    
    self.custom.add(add_quasi_dipole_coordinates, 'modify')
    self.custom.add(add_aacgm_coordinates, 'modify')
    self.custom.add(calculate_ecef_velocity, 'modify')
    self.custom.add(add_sc_attitude_vectors, 'modify')
    self.custom.add(add_iri_thermal_plasma, 'modify')
    self.custom.add(add_hwm_winds_and_ecef_vectors, 'modify')
    self.custom.add(add_igrf, 'modify')
    # project simulated vectors onto s/c basis
    # IGRF
    # create metadata to be added along with vector projection
    in_meta = {'desc':'IGRF geomagnetic field expressed in the s/c basis.',
               'units':'nT'}
    # project IGRF
    self.custom.add(project_ecef_vector_onto_sc, 'modify', 'end', 'B_ecef_x', 'B_ecef_y', 'B_ecef_z',
                                                           'B_sc_x', 'B_sc_y', 'B_sc_z',
                                                           meta=[in_meta.copy(), in_meta.copy(), in_meta.copy()])
    # project total wind vector
    self.custom.add(project_hwm_onto_sc, 'modify')
    # neutral parameters
    self.custom.add(add_msis, 'modify')