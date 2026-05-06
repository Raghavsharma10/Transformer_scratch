def CSPA(hdf5_file_name, cluster_runs, verbose = False, N_clusters_max = None):
    """Cluster-based Similarity Partitioning Algorithm for a consensus function.
    
    Parameters
    ----------
    hdf5_file_name : file handle or string
    
    cluster_runs : array of shape (n_partitions, n_samples)
    
    verbose : bool, optional (default = False)
    
    N_clusters_max : int, optional (default = None)
    
    Returns
    -------
    A vector specifying the cluster label to which each sample has been assigned
    by the CSPA heuristics for consensus clustering.

    Reference
    ---------
    A. Strehl and J. Ghosh, "Cluster Ensembles - A Knowledge Reuse Framework
    for Combining Multiple Partitions".
    In: Journal of Machine Learning Research, 3, pp. 583-617. 2002
    """

    print('*****')
    print("INFO: Cluster_Ensembles: CSPA: consensus clustering using CSPA.")

    if N_clusters_max == None:
        N_clusters_max = int(np.nanmax(cluster_runs)) + 1

    N_runs = cluster_runs.shape[0]
    N_samples = cluster_runs.shape[1]
    if N_samples > 20000:
        raise ValueError("\nERROR: Cluster_Ensembles: CSPA: cannot efficiently "
                         "deal with too large a number of cells.")

    hypergraph_adjacency = load_hypergraph_adjacency(hdf5_file_name)

    s = scipy.sparse.csr_matrix.dot(hypergraph_adjacency.transpose().tocsr(), hypergraph_adjacency)
    s = np.squeeze(np.asarray(s.todense()))
    
    del hypergraph_adjacency
    gc.collect()

    checks(np.divide(s, float(N_runs)), verbose)

    e_sum_before = s.sum()
    sum_after = 100000000.0  
    scale_factor = sum_after / float(e_sum_before)

    with tables.open_file(hdf5_file_name, 'r+') as fileh:
        atom = tables.Float32Atom()
        FILTERS = get_compression_filter(4 * (N_samples ** 2))

        S = fileh.create_carray(fileh.root.consensus_group, 'similarities_CSPA', atom,
                               (N_samples, N_samples), "Matrix of similarities arising "
                               "in Cluster-based Similarity Partitioning", 
                               filters = FILTERS)

        expr = tables.Expr("s * scale_factor")
        expr.set_output(S)
        expr.eval()

        chunks_size = get_chunk_size(N_samples, 3)
        for i in range(0, N_samples, chunks_size):
            tmp = S[i:min(i+chunks_size, N_samples)]
            S[i:min(i+chunks_size, N_samples)] = np.rint(tmp)

    return metis(hdf5_file_name, N_clusters_max)