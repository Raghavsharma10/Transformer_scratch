def load_hypergraph_adjacency(hdf5_file_name):
    """
    
    Parameters
    ----------
    hdf5_file_name : file handle or string
    
    Returns
    -------
    hypergraph_adjacency : compressed sparse row matrix
    """

    with tables.open_file(hdf5_file_name, 'r+') as fileh:
        pars = []
        for par in ('data', 'indices', 'indptr', 'shape'):
            pars.append(getattr(fileh.root.consensus_group, par).read())

    hypergraph_adjacency = scipy.sparse.csr_matrix(tuple(pars[:3]), shape = pars[3])
    
    return hypergraph_adjacency