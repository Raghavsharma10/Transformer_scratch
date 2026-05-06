def cluster_ensembles(cluster_runs, hdf5_file_name = None, verbose = False, N_clusters_max = None):
    """Call up to three different functions for heuristic ensemble clustering
       (namely CSPA, HGPA and MCLA) then select as the definitive
       consensus clustering the one with the highest average mutual information score 
       between its vector of consensus labels and the vectors of labels associated to each
       partition from the ensemble.

    Parameters
    ----------
    cluster_runs : array of shape (n_partitions, n_samples)
        Each row of this matrix is such that the i-th entry corresponds to the
        cluster ID to which the i-th sample of the data-set has been classified
        by this particular clustering. Samples not selected for clustering
        in a given round are are tagged by an NaN.
        
    hdf5_file_name : file object or string, optional (default = None)
        The handle or name of an HDF5 file where any array needed
        for consensus_clustering and too large to fit into memory 
        is to be stored. Created if not specified at input.
        
    verbose : Boolean, optional (default = False)
        Specifies if messages concerning the status of the many functions 
        subsequently called 'cluster_ensembles' will be displayed
        on the standard output.

    N_clusters_max : int, optional
        The number of clusters in which to partition the samples into 
        a consensus clustering. This defaults to the highest number of clusters
        encountered in the sets of independent clusterings on subsamples 
        of the data-set (i.e. the maximum of the entries in "cluster_runs").

    Returns
    -------
    cluster_ensemble : array of shape (n_samples,)
        For the final ensemble clustering, this vector contains the 
        cluster IDs of each sample in the whole data-set.

    Reference
    ---------
    A. Strehl and J. Ghosh, "Cluster Ensembles - A Knowledge Reuse Framework
    for Combining Multiple Partitions".
    In: Journal of Machine Learning Research, 3, pp. 583-617. 2002  
    """

    if hdf5_file_name is None:
        hdf5_file_name = './Cluster_Ensembles.h5'
    fileh = tables.open_file(hdf5_file_name, 'w')
    fileh.create_group(fileh.root, 'consensus_group')
    fileh.close()

    cluster_ensemble = []
    score = np.empty(0)

    if cluster_runs.shape[1] > 10000:
        consensus_functions = [HGPA, MCLA]
        function_names = ['HGPA', 'MCLA']
        print("\nINFO: Cluster_Ensembles: cluster_ensembles: "
              "due to a rather large number of cells in your data-set, "
              "using only 'HyperGraph Partitioning Algorithm' (HGPA) "
              "and 'Meta-CLustering Algorithm' (MCLA) "
              "as ensemble consensus functions.\n")
    else:
        consensus_functions = [CSPA, HGPA, MCLA]
        function_names = ['CSPA', 'HGPA', 'MCLA']

    hypergraph_adjacency = build_hypergraph_adjacency(cluster_runs)
    store_hypergraph_adjacency(hypergraph_adjacency, hdf5_file_name)

    for i in range(len(consensus_functions)):
        cluster_ensemble.append(consensus_functions[i](hdf5_file_name, cluster_runs, verbose, N_clusters_max))
        score = np.append(score, ceEvalMutual(cluster_runs, cluster_ensemble[i], verbose))
        print("\nINFO: Cluster_Ensembles: cluster_ensembles: "
              "{0} at {1}.".format(function_names[i], score[i]))
        print('*****')

    return cluster_ensemble[np.argmax(score)]