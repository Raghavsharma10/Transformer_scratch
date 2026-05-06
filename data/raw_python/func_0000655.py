def store_hypergraph_adjacency(hypergraph_adjacency, hdf5_file_name):
    """Write an hypergraph adjacency to disk to disk in an HDF5 data structure.
    
    Parameters
    ----------
    hypergraph_adjacency : compressed sparse row matrix
    
    hdf5_file_name : file handle or string
    """
   
    assert(hypergraph_adjacency.__class__ == scipy.sparse.csr.csr_matrix)
    
    byte_counts = hypergraph_adjacency.data.nbytes + hypergraph_adjacency.indices.nbytes + hypergraph_adjacency.indptr.nbytes
    FILTERS = get_compression_filter(byte_counts)

    with tables.open_file(hdf5_file_name, 'r+') as fileh:
        for par in ('data', 'indices', 'indptr', 'shape'):
            try:
                n = getattr(fileh.root.consensus_group, par)
                n._f_remove()
            except AttributeError:
                pass

            array = np.array(getattr(hypergraph_adjacency, par))

            atom = tables.Atom.from_dtype(array.dtype)
            ds = fileh.create_carray(fileh.root.consensus_group, par, atom, 
                                     array.shape, filters = FILTERS)

            ds[:] = array