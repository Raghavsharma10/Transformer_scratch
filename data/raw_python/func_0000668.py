def wgraph(hdf5_file_name, w = None, method = 0):
    """Write a graph file in a format apposite to later use by METIS or HMETIS.
    
    Parameters
    ----------
    hdf5_file_name : file handle or string
    
    w : list or array, optional (default = None)
    
    method : int, optional (default = 0)
    
    Returns
    -------
    file_name : string
    """

    print('\n#')

    if method == 0:
        fileh = tables.open_file(hdf5_file_name, 'r+')
        e_mat = fileh.root.consensus_group.similarities_CSPA
        file_name = 'wgraph_CSPA'
    elif method == 1:
        fileh = tables.open_file(hdf5_file_name, 'r+')
        e_mat = fileh.root.consensus_group.similarities_MCLA
        file_name = 'wgraph_MCLA'
    elif method in {2, 3}:
        hypergraph_adjacency = load_hypergraph_adjacency(hdf5_file_name)
        e_mat = hypergraph_adjacency.copy().transpose()
        file_name = 'wgraph_HGPA'
        fileh = tables.open_file(hdf5_file_name, 'r+')
    else:
        raise ValueError("\nERROR: Cluster_Ensembles: wgraph: "
                         "invalid code for choice of method; "
                         "choose either 0, 1, 2 or 3.")

    if w is None:
        w = []

    N_rows = e_mat.shape[0]
    N_cols = e_mat.shape[1]

    if method in {0, 1}:
        diag_ind = np.diag_indices(N_rows)
        e_mat[diag_ind] = 0

    if method == 1:
        scale_factor = 100.0
        w_sum_before = np.sum(w)
        w *= scale_factor
        w = np.rint(w)

    with open(file_name, 'w') as file:
        print("INFO: Cluster_Ensembles: wgraph: writing {}.".format(file_name))

        if method == 0:
            sz = float(np.sum(e_mat[:] > 0)) / 2
            if int(sz) == 0:
                return 'DO_NOT_PROCESS'
            else:
                file.write('{} {} 1\n'.format(N_rows, int(sz)))
        elif method == 1:
            chunks_size = get_chunk_size(N_cols, 2)
            N_chunks, remainder = divmod(N_rows, chunks_size)
            if N_chunks == 0:
                sz = float(np.sum(e_mat[:] > 0)) / 2
            else:
                sz = 0
                for i in range(N_chunks):
                    M = e_mat[i*chunks_size:(i+1)*chunks_size]
                    sz += float(np.sum(M > 0))
                if remainder != 0:
                    M = e_mat[N_chunks*chunks_size:N_rows]
                    sz += float(np.sum(M > 0))
                sz = float(sz) / 2 
            file.write('{} {} 11\n'.format(N_rows, int(sz)))
        else:
            file.write('{} {} 1\n'.format(N_cols, N_rows))
                    
        if method in {0, 1}:
            chunks_size = get_chunk_size(N_cols, 2)
            for i in range(0, N_rows, chunks_size):
                M = e_mat[i:min(i+chunks_size, N_rows)]

                for j in range(M.shape[0]):
                    edges = np.where(M[j] > 0)[0]
                    weights = M[j, edges]

                    if method == 0:
                        interlaced = np.zeros(2 * edges.size, dtype = int)
                        # METIS and hMETIS have vertices numbering starting from 1:
                        interlaced[::2] = edges + 1 
                        interlaced[1::2] = weights
                    else:
                        interlaced = np.zeros(1 + 2 * edges.size, dtype = int)
                        interlaced[0] = w[i + j]
                        # METIS and hMETIS have vertices numbering starting from 1:
                        interlaced[1::2] = edges + 1 
                        interlaced[2::2] = weights

                    for elt in interlaced:
                        file.write('{} '.format(int(elt)))
                    file.write('\n')  
        else:
            print("INFO: Cluster_Ensembles: wgraph: {N_rows} vertices and {N_cols} "
                  "non-zero hyper-edges.".format(**locals()))

            chunks_size = get_chunk_size(N_rows, 2)
            for i in range(0, N_cols, chunks_size):
                M = np.asarray(e_mat[:, i:min(i+chunks_size, N_cols)].todense())
                for j in range(M.shape[1]):
                    edges = np.where(M[:, j] > 0)[0]
                    if method == 2:
                        weight = np.array(M[:, j].sum(), dtype = int)
                    else:
                        weight = w[i + j]
                    # METIS and hMETIS require vertices numbering starting from 1:
                    interlaced = np.append(weight, edges + 1) 
               
                    for elt in interlaced:
                        file.write('{} '.format(int(elt)))
                    file.write('\n')
    
    if method in {0, 1}:
        fileh.remove_node(fileh.root.consensus_group, e_mat.name)

    fileh.close()

    print('#')

    return file_name