def boolMasksToImage(masks):
    '''
    Transform at maximum 8 bool layers --> 2d arrays, dtype=(bool,int)
    to one 8bit image
    '''
    assert len(masks) <= 8, 'can only transform up to 8 masks into image'
    masks = np.asarray(masks, dtype=np.uint8)
    assert masks.ndim == 3, 'layers need to be stack of 2d arrays'
    return np.packbits(masks, axis=0)[0].T