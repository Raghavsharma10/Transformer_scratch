def timeseries_from_mat(filename, varname=None, fs=1.0):
    """load a multi-channel Timeseries from a MATLAB .mat file

    Args:
      filename (str): .mat file to load
      varname (str): variable name. only needed if there is more than one
        variable saved in the .mat file
      fs (scalar): sample rate of timeseries in Hz. (constant timestep assumed)

    Returns:
      Timeseries
    """
    import scipy.io as sio
    if varname is None:
        mat_dict = sio.loadmat(filename)
        if len(mat_dict) > 1:
            raise ValueError('Must specify varname: file contains '
                             'more than one variable. ')
    else:
        mat_dict = sio.loadmat(filename, variable_names=(varname,))
        array = mat_dict.popitem()[1]
    return Timeseries(array, fs=fs)