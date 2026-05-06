def save_mat(ts, filename):
    """save a Timeseries to a MATLAB .mat file
    Args:
      ts (Timeseries): the timeseries to save
      filename (str): .mat filename to save to
    """
    import scipy.io as sio
    tspan = ts.tspan
    fs = (1.0*len(tspan) - 1) / (tspan[-1] - tspan[0])
    mat_dict = {'data': np.asarray(ts),
                'fs': fs,
                'labels': ts.labels[1]}
    sio.savemat(filename, mat_dict, do_compression=True)
    return