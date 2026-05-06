def identity_on_pols(self, context):
    """
    Returns [[1, 0], tiled up to other dimensions
             [0, 1]]
    """
    A = np.empty(context.shape, context.dtype)
    A[:,:,:] = [[[1,0,0,1]]]
    return A