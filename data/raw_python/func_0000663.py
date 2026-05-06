def HGPA(hdf5_file_name, cluster_runs, verbose = False, N_clusters_max = None):
    """HyperGraph-Partitioning Algorithm for a consensus function.
    
    Parameters
    ----------
    hdf5_file_name : string or file handle
    
    cluster_runs: array of shape (n_partitions, n_samples)
    
    verbose : bool, optional (default = False)
    
    N_clusters_max : int, optional (default = None)
    
    Returns
    -------
    A vector specifying the cluster label to which each sample has been assigned
    by the HGPA approximation algorithm for consensus clustering.

    Reference
    ---------
    A. Strehl and J. Ghosh, "Cluster Ensembles - A Knowledge Reuse Framework
    for Combining Multiple Partitions".
    In: Journal of Machine Learning Research, 3, pp. 583-617. 2002
    """
    
    print('\n*****')
    print("INFO: Cluster_Ensembles: HGPA: consensus clustering using HGPA.")

    if N_clusters_max == None:
        N_clusters_max = int(np.nanmax(cluster_runs)) + 1

    return hmetis(hdf5_file_name, N_clusters_max)