def write_gdf(gdf, fname):
    """
    Fast line-by-line gdf-file write function
    
    
    Parameters
    ----------
    gdf : numpy.ndarray
        Column 0 is gids, columns 1: are values.
    fname : str
        Path to gdf-file.
    
    
    Returns
    -------
    None
    
    """
    gdf_file = open(fname, 'w')
    for line in gdf:
        for i in np.arange(len(line)):
            gdf_file.write(str(line[i]) + '\t')
        gdf_file.write('\n')
    
    return None