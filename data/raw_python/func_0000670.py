def overlap_matrix(hdf5_file_name, consensus_labels, cluster_runs):
    """Writes on disk (in an HDF5 file whose handle is provided as the first
       argument to this function) a stack of matrices, each describing
       for a particular run the overlap of cluster ID's that are matching 
       each of the cluster ID's stored in 'consensus_labels' 
       (the vector of labels obtained by ensemble clustering). 
       Returns also the adjacency matrix for consensus clustering 
       and a vector of mutual informations between each of the clusterings 
       from the ensemble and their consensus.
       
    Parameters
    ----------
    hdf5_file_name : file handle or string
    
    consensus_labels : array of shape (n_samples,)
    
    cluster_runs : array of shape (n_partitions, n_samples)
    
    Returns
    -------
    cluster_dims_list : 
    
    mutual_info_list :
    
    consensus_adjacency :
    """

    if reduce(operator.mul, cluster_runs.shape, 1) == max(cluster_runs.shape):
        cluster_runs = cluster_runs.reshape(1, -1)

    N_runs, N_samples = cluster_runs.shape
    N_consensus_labels = np.unique(consensus_labels).size

    indices_consensus_adjacency = np.empty(0, dtype = np.int32)
    indptr_consensus_adjacency = np.zeros(1, dtype = np.int64)

    for k in range(N_consensus_labels):
        indices_consensus_adjacency = np.append(indices_consensus_adjacency, np.where(consensus_labels == k)[0])
        indptr_consensus_adjacency = np.append(indptr_consensus_adjacency, indices_consensus_adjacency.size)

    data_consensus_adjacency = np.ones(indices_consensus_adjacency.size, dtype = int) 

    consensus_adjacency = scipy.sparse.csr_matrix((data_consensus_adjacency, indices_consensus_adjacency, indptr_consensus_adjacency), 
                                                  shape = (N_consensus_labels, N_samples))

    fileh = tables.open_file(hdf5_file_name, 'r+')
    
    FILTERS = get_compression_filter(4 * N_consensus_labels * N_runs)

    overlap_matrix = fileh.create_earray(fileh.root.consensus_group, 'overlap_matrix',
                                         tables.Float32Atom(), (0, N_consensus_labels), 
                                         "Matrix of overlaps between each run and "
                                         "the consensus labellings", filters = FILTERS,
                                         expectedrows = N_consensus_labels * N_runs)

    mutual_info_list = []
    cluster_dims_list =  [0]

    for i in range(N_runs):
        M = cluster_runs[i]

        mutual_info_list.append(ceEvalMutual(M, consensus_labels))

        finite_indices = np.where(np.isfinite(M))[0]
        positive_indices = np.where(M >= 0)[0]
        selected_indices = np.intersect1d(finite_indices, positive_indices, assume_unique = True)
        cluster_ids = np.unique(M[selected_indices])
        n_ids = cluster_ids.size

        cluster_dims_list.append(n_ids)

        unions = np.zeros((n_ids, N_consensus_labels), dtype = float)

        indices = np.empty(0, dtype = int)
        indptr = [0]

        c = 0
        for elt in cluster_ids:
            indices = np.append(indices, np.where(M == elt)[0])
            indptr.append(indices.size)

            for k in range(N_consensus_labels):
                x = indices_consensus_adjacency[indptr_consensus_adjacency[k]:indptr_consensus_adjacency[k+1]]
                unions[c, k] = np.union1d(indices, x).size 
 
            c += 1 

        data = np.ones(indices.size, dtype = int)
    
        I = scipy.sparse.csr_matrix((data, indices, indptr), shape = (n_ids, N_samples))

        intersections = I.dot(consensus_adjacency.transpose())
        intersections = np.squeeze(np.asarray(intersections.todense()))

        overlap_matrix.append(np.divide(intersections, unions))

    fileh.close()

    return cluster_dims_list, mutual_info_list, consensus_adjacency