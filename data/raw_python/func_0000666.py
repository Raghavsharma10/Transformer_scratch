def metis(hdf5_file_name, N_clusters_max):
    """METIS algorithm by Karypis and Kumar. Partitions the induced similarity graph 
        passed by CSPA.

    Parameters
    ----------
    hdf5_file_name : string or file handle
    
    N_clusters_max : int
    
    Returns
    -------
    labels : array of shape (n_samples,)
        A vector of labels denoting the cluster to which each sample has been assigned
        as a result of the CSPA heuristics for consensus clustering.
    
    Reference
    ---------
    G. Karypis and V. Kumar, "A Fast and High Quality Multilevel Scheme for
    Partitioning Irregular Graphs"
    In: SIAM Journal on Scientific Computing, Vol. 20, No. 1, pp. 359-392, 1999.
    """

    file_name = wgraph(hdf5_file_name)
    labels = sgraph(N_clusters_max, file_name)
    subprocess.call(['rm', file_name])

    return labels