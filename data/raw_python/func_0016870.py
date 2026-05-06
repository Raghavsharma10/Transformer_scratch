def default_stokes(self, context):
    """
    Returns [[1, 0], tiled up to other dimensions
             [0, 0]]
    """
    A = np.empty(context.shape, context.dtype)
    A[:,:,:] = [[[1,0,0,0]]]
    return A