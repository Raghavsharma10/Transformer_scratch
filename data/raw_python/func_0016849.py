def _setup_hypercube(cube, slvr_cfg):
    """ Sets up the hypercube given a solver configuration """
    mbu.register_default_dimensions(cube, slvr_cfg)

    # Configure the dimensions of the beam cube
    cube.register_dimension('beam_lw', 2,
                            description='E Beam cube l width')

    cube.register_dimension('beam_mh', 2,
                            description='E Beam cube m height')

    cube.register_dimension('beam_nud', 2,
                            description='E Beam cube nu depth')

    # =========================================
    # Register hypercube Arrays and Properties
    # =========================================

    from montblanc.impl.rime.tensorflow.config import (A, P)

    def _massage_dtypes(A, T):
        def _massage_dtype_in_dict(D):
            new_dict = D.copy()
            new_dict['dtype'] = mbu.dtype_from_str(D['dtype'], T)
            return new_dict

        return [_massage_dtype_in_dict(D) for D in A]

    dtype = slvr_cfg['dtype']
    is_f32 = dtype == 'float'

    T = {
        'ft' : np.float32 if is_f32 else np.float64,
        'ct' : np.complex64 if is_f32 else np.complex128,
        'int' : int,
    }

    cube.register_properties(_massage_dtypes(P, T))
    cube.register_arrays(_massage_dtypes(A, T))