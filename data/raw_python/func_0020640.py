def find_node_match(simple_vertices, complex_vertices):
    """
    Thanks to juhuntenburg.
    Functions taken from https://github.com/juhuntenburg/brainsurfacescripts

    Finds those points on the complex mesh that correspond best to the
    simple mesh while forcing a one-to-one mapping.
    """

    import scipy.spatial

    # make array for writing in final voronoi seed indices
    voronoi_seed_idx = np.zeros((simple_vertices.shape[0],), dtype='int64')-1
    missing = np.where(voronoi_seed_idx == -1)[0].shape[0]
    mapping_single = np.zeros_like(voronoi_seed_idx)

    neighbours = 0
    col = 0

    while missing != 0:

        neighbours += 100
        # find nearest neighbours
        inaccuracy, mapping = scipy.spatial.KDTree(
            complex_vertices).query(simple_vertices, k=neighbours)
        # go through columns of nearest neighbours until unique mapping is
        # achieved, if not before end of neighbours, extend number of
        # neighbours
        while col < neighbours:
            # find all missing voronoi seed indices
            missing_idx = np.where(voronoi_seed_idx == -1)[0]
            missing = missing_idx.shape[0]
            if missing == 0:
                break
            else:
                # for missing entries fill in next neighbour
                mapping_single[missing_idx] = np.copy(
                    mapping[missing_idx, col])
                # find unique values in mapping_single
                unique, double_idx = np.unique(
                    mapping_single, return_inverse=True)
                # empty voronoi seed index
                voronoi_seed_idx = np.zeros(
                    (simple_vertices.shape[0],), dtype='int64')-1
                # fill voronoi seed idx with unique values
                for u in range(unique.shape[0]):
                    # find the indices of this value in mapping
                    entries = np.where(double_idx == u)[0]
                    # set the first entry to the value
                    voronoi_seed_idx[entries[0]] = unique[u]
                # go to next column
                col += 1

    return voronoi_seed_idx, inaccuracy