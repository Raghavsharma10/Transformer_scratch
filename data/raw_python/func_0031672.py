def read_gdf(fname):
    """
    Fast line-by-line gdf-file reader.
    
    
    Parameters
    ----------
    fname : str 
        Path to gdf-file.
    
    
    Returns
    -------
    numpy.ndarray
        ([gid, val0, val1, **]), dtype=object) mixed datatype array
            
    """
    
    gdf_file = open(fname, 'r')
    gdf = []
    for l in gdf_file:
        data = l.split()
        gdf += [data]

    gdf = np.array(gdf, dtype=object)
    
    if gdf.size > 0:
        gdf[:, 0] = gdf[:, 0].astype(int)
        gdf[:, 1:] = gdf[:, 1:].astype(float)
    
    return np.array(gdf)