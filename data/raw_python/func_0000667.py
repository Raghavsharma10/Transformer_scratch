def hmetis(hdf5_file_name, N_clusters_max, w = None):
    """Gives cluster labels ranging from 1 to N_clusters_max for 
        hypergraph partitioning required for HGPA.

    Parameters
    ----------
    hdf5_file_name : file handle or string
    
    N_clusters_max : int
    
    w : array, optional (default = None)
    
    Returns
    -------
    labels : array of shape (n_samples,)
        A vector of labels denoting the cluster to which each sample has been assigned
        as a result of the HGPA approximation algorithm for consensus clustering.
    
    Reference
    ---------
    G. Karypis, R. Aggarwal, V. Kumar and S. Shekhar, "Multilevel hypergraph
    partitioning: applications in VLSI domain" 
    In: IEEE Transactions on Very Large Scale Integration (VLSI) Systems, 
    Vol. 7, No. 1, pp. 69-79, 1999.
    """

    if w is None:
        file_name = wgraph(hdf5_file_name, None, 2)
    else:
        file_name = wgraph(hdf5_file_name, w, 3)
    labels = sgraph(N_clusters_max, file_name)
    labels = one_to_max(labels)

    subprocess.call(['rm', file_name])

    return labels