def grid_edges(shape, inds=None, return_directions=True):
    """
    Get list of grid edges
    :param shape:
    :param inds:
    :param return_directions:
    :return:
    """
    if inds is None:
        inds = np.arange(np.prod(shape)).reshape(shape)
    # if not self.segparams['use_boundary_penalties'] and \
    #         boundary_penalties_fcn is None :
    if len(shape) == 2:
        edgx = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
        edgy = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]

        edges = [edgx, edgy]

        directions = [
            np.ones([edgx.shape[0]], dtype=np.int8) * 0,
            np.ones([edgy.shape[0]], dtype=np.int8) * 1,
        ]

    elif len(shape) == 3:
        # This is faster for some specific format
        edgx = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
        edgy = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
        edgz = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]
        edges = [edgx, edgy, edgz]
    else:
        logger.error("Expected 2D or 3D data")

    # for all edges along first direction put 0, for second direction put 1, for third direction put 3
    if return_directions:
        directions = []
        for idirection in range(len(shape)):
            directions.append(
                np.ones([edges[idirection].shape[0]], dtype=np.int8) * idirection
            )
    edges = np.concatenate(edges)
    if return_directions:
        edge_dir = np.concatenate(directions)
        return edges, edge_dir
    else:
        return edges