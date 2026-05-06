def load_h5_data(path='', data_type='LFP', y=None, electrode=None,
                 warmup=0., scaling=1.):
    """
    Function loading results from hdf5 file
    
    
    Parameters
    ----------
    path : str
        Path to hdf5-file
    data_type : str
        Signal types in ['CSD' , 'LFP', 'CSDsum', 'LFPsum'].
    y : None or str
        Name of population.
    electrode : None or int
        TODO: update, electrode is NOT USED
    warmup : float
        Lower cutoff of time series to remove possible transients
    scaling : float,
        Scaling factor for population size that determines the amount of loaded
        single-cell signals
                
    
    Returns
    ----------
    numpy.ndarray
        [electrode id, compound signal] if `y` is None
    numpy.ndarray
        [cell id, electrode, single-cell signal] otherwise
    
    """
    assert y is not None or electrode is not None
    
    if y is not None:
        f = h5py.File(os.path.join(path, '%s_%ss.h5' %(y,data_type)))
        data = f['data'].value[:,:, warmup:]
        if scaling != 1.:
            np.random.shuffle(data)
            num_cells = int(len(data)*scaling)
            data = data[:num_cells,:, warmup:]
    else:
        f = h5py.File(os.path.join(path, '%ssum.h5' %data_type))
        data = f['data'].value[:, warmup:]

    return data