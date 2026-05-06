def check_images(data):
    """
    Check and reformat input images if needed
    """
    if isinstance(data, ndarray):
        data = fromarray(data)
    
    if not isinstance(data, Images):
        data = fromarray(asarray(data))

    if len(data.shape) not in set([3, 4]):
        raise Exception('Number of image dimensions %s must be 2 or 3' % (len(data.shape)))

    return data