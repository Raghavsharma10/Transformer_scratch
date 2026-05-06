def MCLA(hdf5_file_name, cluster_runs, verbose = False, N_clusters_max = None):
    """Meta-CLustering Algorithm for a consensus function.
    
    Parameters
    ----------
    hdf5_file_name : file handle or string
    
    cluster_runs : array of shape (n_partitions, n_samples)
    
    verbose : bool, optional (default = False)
    
    N_clusters_max : int, optional (default = None)
    
    Returns
    -------
    A vector specifying the cluster label to which each sample has been assigned
    by the MCLA approximation algorithm for consensus clustering.

    Reference
    ---------
    A. Strehl and J. Ghosh, "Cluster Ensembles - A Knowledge Reuse Framework
    for Combining Multiple Partitions".
    In: Journal of Machine Learning Research, 3, pp. 583-617. 2002
    """

    print('\n*****')
    print('INFO: Cluster_Ensembles: MCLA: consensus clustering using MCLA.')

    if N_clusters_max == None:
        N_clusters_max = int(np.nanmax(cluster_runs)) + 1

    N_runs = cluster_runs.shape[0]
    N_samples = cluster_runs.shape[1]

    print("INFO: Cluster_Ensembles: MCLA: preparing graph for meta-clustering.")

    hypergraph_adjacency = load_hypergraph_adjacency(hdf5_file_name)
    w = hypergraph_adjacency.sum(axis = 1)

    N_rows = hypergraph_adjacency.shape[0]

    print("INFO: Cluster_Ensembles: MCLA: done filling hypergraph adjacency matrix. "
          "Starting computation of Jaccard similarity matrix.")

    # Next, obtain a matrix of pairwise Jaccard similarity scores between the rows of the hypergraph adjacency matrix.
    with tables.open_file(hdf5_file_name, 'r+') as fileh:
        FILTERS = get_compression_filter(4 * (N_rows ** 2))
    
        similarities_MCLA = fileh.create_carray(fileh.root.consensus_group, 
                                   'similarities_MCLA', tables.Float32Atom(), 
                                   (N_rows, N_rows), "Matrix of pairwise Jaccard "
                                   "similarity scores", filters = FILTERS)

        scale_factor = 100.0

        print("INFO: Cluster_Ensembles: MCLA: "
              "starting computation of Jaccard similarity matrix.")

        squared_MCLA = hypergraph_adjacency.dot(hypergraph_adjacency.transpose())

        squared_sums = hypergraph_adjacency.sum(axis = 1)
        squared_sums = np.squeeze(np.asarray(squared_sums))

        chunks_size = get_chunk_size(N_rows, 7)
        for i in range(0, N_rows, chunks_size):
            n_dim = min(chunks_size, N_rows - i)

            temp = squared_MCLA[i:min(i+chunks_size, N_rows), :].todense()
            temp = np.squeeze(np.asarray(temp))

            x = squared_sums[i:min(i+chunks_size, N_rows)]
            x = x.reshape(-1, 1)
            x = np.dot(x, np.ones((1, squared_sums.size)))

            y = np.dot(np.ones((n_dim, 1)), squared_sums.reshape(1, -1))
        
            temp = np.divide(temp, x + y - temp)
            temp *= scale_factor

            Jaccard_matrix = np.rint(temp)
            similarities_MCLA[i:min(i+chunks_size, N_rows)] = Jaccard_matrix

            del Jaccard_matrix, temp, x, y
            gc.collect()
 
    # Done computing the matrix of pairwise Jaccard similarity scores.
    print("INFO: Cluster_Ensembles: MCLA: done computing the matrix of "
          "pairwise Jaccard similarity scores.")

    cluster_labels = cmetis(hdf5_file_name, N_clusters_max, w)
    cluster_labels = one_to_max(cluster_labels)
    # After 'cmetis' returns, we are done with clustering hyper-edges

    # We are now ready to start the procedure meant to collapse meta-clusters.
    N_consensus = np.amax(cluster_labels) + 1

    fileh = tables.open_file(hdf5_file_name, 'r+')

    FILTERS = get_compression_filter(4 * N_consensus * N_samples)
    
    clb_cum = fileh.create_carray(fileh.root.consensus_group, 'clb_cum', 
                                  tables.Float32Atom(), (N_consensus, N_samples), 
                                  'Matrix of mean memberships, forming meta-clusters', 
                                  filters = FILTERS)  
 
    chunks_size = get_chunk_size(N_samples, 7)
    for i in range(0, N_consensus, chunks_size):
        x = min(chunks_size, N_consensus - i)
        matched_clusters = np.where(cluster_labels == np.reshape(np.arange(i, min(i + chunks_size, N_consensus)), newshape = (x, 1)))
        M = np.zeros((x, N_samples))
        for j in range(x):
            coord = np.where(matched_clusters[0] == j)[0]
            M[j] = np.asarray(hypergraph_adjacency[matched_clusters[1][coord], :].mean(axis = 0))
        clb_cum[i:min(i+chunks_size, N_consensus)] = M
    
    # Done with collapsing the hyper-edges into a single meta-hyper-edge, 
    # for each of the (N_consensus - 1) meta-clusters.

    del hypergraph_adjacency
    gc.collect()

    # Each object will now be assigned to its most associated meta-cluster.
    chunks_size = get_chunk_size(N_consensus, 4)
    N_chunks, remainder = divmod(N_samples, chunks_size)
    if N_chunks == 0:
        null_columns = np.where(clb_cum[:].sum(axis = 0) == 0)[0]
    else:
        szumsz = np.zeros(0)
        for i in range(N_chunks):
            M = clb_cum[:, i*chunks_size:(i+1)*chunks_size]
            szumsz = np.append(szumsz, M.sum(axis = 0))
        if remainder != 0:
            M = clb_cum[:, N_chunks*chunks_size:N_samples]
            szumsz = np.append(szumsz, M.sum(axis = 0))
        null_columns = np.where(szumsz == 0)[0]

    if null_columns.size != 0:
        print("INFO: Cluster_Ensembles: MCLA: {} objects with all zero associations "
              "in 'clb_cum' matrix of meta-clusters.".format(null_columns.size))
        clb_cum[:, null_columns] = np.random.rand(N_consensus, null_columns.size)

    random_state = np.random.RandomState()

    tmp = fileh.create_carray(fileh.root.consensus_group, 'tmp', tables.Float32Atom(),
                              (N_consensus, N_samples), "Temporary matrix to help with "
                              "collapsing to meta-hyper-edges", filters = FILTERS)

    chunks_size = get_chunk_size(N_samples, 2)
    N_chunks, remainder = divmod(N_consensus, chunks_size)
    if N_chunks == 0:
        tmp[:] = random_state.rand(N_consensus, N_samples)
    else:
        for i in range(N_chunks):
            tmp[i*chunks_size:(i+1)*chunks_size] = random_state.rand(chunks_size, N_samples)
        if remainder !=0:
            tmp[N_chunks*chunks_size:N_consensus] = random_state.rand(remainder, N_samples)

    expr = tables.Expr("clb_cum + (tmp / 10000)")
    expr.set_output(clb_cum)
    expr.eval()

    expr = tables.Expr("abs(tmp)")
    expr.set_output(tmp)
    expr.eval()

    chunks_size = get_chunk_size(N_consensus, 2)
    N_chunks, remainder = divmod(N_samples, chunks_size)
    if N_chunks == 0:
        sum_diag = tmp[:].sum(axis = 0)
    else:
        sum_diag = np.empty(0)
        for i in range(N_chunks):
            M = tmp[:, i*chunks_size:(i+1)*chunks_size]
            sum_diag = np.append(sum_diag, M.sum(axis = 0))
        if remainder != 0:
            M = tmp[:, N_chunks*chunks_size:N_samples]
            sum_diag = np.append(sum_diag, M.sum(axis = 0))

    fileh.remove_node(fileh.root.consensus_group, "tmp") 
    # The corresponding disk space will be freed after a call to 'fileh.close()'.

    inv_sum_diag = np.reciprocal(sum_diag.astype(float))

    if N_chunks == 0:
        clb_cum *= inv_sum_diag
        max_entries = np.amax(clb_cum, axis = 0)
    else:
        max_entries = np.zeros(N_samples)
        for i in range(N_chunks):
            clb_cum[:, i*chunks_size:(i+1)*chunks_size] *= inv_sum_diag[i*chunks_size:(i+1)*chunks_size]
            max_entries[i*chunks_size:(i+1)*chunks_size] = np.amax(clb_cum[:, i*chunks_size:(i+1)*chunks_size], axis = 0)
        if remainder != 0:
            clb_cum[:, N_chunks*chunks_size:N_samples] *= inv_sum_diag[N_chunks*chunks_size:N_samples]
            max_entries[N_chunks*chunks_size:N_samples] = np.amax(clb_cum[:, N_chunks*chunks_size:N_samples], axis = 0)

    cluster_labels = np.zeros(N_samples, dtype = int)
    winner_probabilities = np.zeros(N_samples)
    
    chunks_size = get_chunk_size(N_samples, 2)
    for i in reversed(range(0, N_consensus, chunks_size)):
        ind = np.where(np.tile(max_entries, (min(chunks_size, N_consensus - i), 1)) == clb_cum[i:min(i+chunks_size, N_consensus)])
        cluster_labels[ind[1]] = i + ind[0]
        winner_probabilities[ind[1]] = clb_cum[(ind[0] + i, ind[1])]       

    # Done with competing for objects.

    cluster_labels = one_to_max(cluster_labels)

    print("INFO: Cluster_Ensembles: MCLA: delivering "
          "{} clusters.".format(np.unique(cluster_labels).size))
    print("INFO: Cluster_Ensembles: MCLA: average posterior "
          "probability is {}".format(np.mean(winner_probabilities)))
    if cluster_labels.size <= 7:
        print("INFO: Cluster_Ensembles: MCLA: the winning posterior probabilities are:")
        print(winner_probabilities)
        print("'INFO: Cluster_Ensembles: MCLA: the full posterior probabilities are:")
        print(clb_cum)

    fileh.remove_node(fileh.root.consensus_group, "clb_cum")
    fileh.close()

    return cluster_labels