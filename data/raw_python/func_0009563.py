def imageToBoolMasks(arr):
    '''inverse of [boolMasksToImage]'''
    assert arr.dtype == np.uint8, 'image needs to be dtype=uint8'
    masks = np.unpackbits(arr).reshape(*arr.shape, 8)
    return np.swapaxes(masks, 2, 0)