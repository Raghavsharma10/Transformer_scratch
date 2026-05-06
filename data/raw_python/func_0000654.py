def build_hypergraph_adjacency(cluster_runs):
    """Return the adjacency matrix to a hypergraph, in sparse matrix representation.
    
    Parameters
    ----------
    cluster_runs : array of shape (n_partitions, n_samples)
    
    Returns
    -------
    hypergraph_adjacency : compressed sparse row matrix
        Represents the hypergraph associated with an ensemble of partitions,
        each partition corresponding to a row of the array 'cluster_runs'
        provided at input.
    """

    N_runs = cluster_runs.shape[0]

    hypergraph_adjacency = create_membership_matrix(cluster_runs[0])
    for i in range(1, N_runs):
        hypergraph_adjacency = scipy.sparse.vstack([hypergraph_adjacency,
                                                   create_membership_matrix(cluster_runs[i])], 
                                                   format = 'csr')

    return hypergraph_adjacency