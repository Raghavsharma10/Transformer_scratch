def polylinesFromBinImage(img, minimum_cluster_size=6,
                          remove_small_obj_size=3,
                          reconnect_size=3,
                          max_n_contours=None, max_len_contour=None,
                          copy=True):
    '''
    return a list of arrays of un-branching contours

    img -> (boolean) array 

    optional:
    ---------
    minimum_cluster_size -> minimum number of pixels connected together to build a contour

    ##search_kernel_size -> TODO
    ##min_search_kernel_moment -> TODO

    numeric:
    -------------
    max_n_contours -> maximum number of possible contours in img
    max_len_contour -> maximum contour length

    '''
    assert minimum_cluster_size > 1
    assert reconnect_size % 2, 'ksize needs to be odd'

    # assert search_kernel_size == 0 or search_kernel_size > 2 and search_kernel_size%2, 'kernel size needs to be odd'
    # assume array size parameters, is not given:
    if max_n_contours is None:
        max_n_contours = max(img.shape)
    if max_len_contour is None:
        max_len_contour = sum(img.shape[:2])
    # array containing coord. of all contours:
    contours = np.zeros(shape=(max_n_contours, max_len_contour, 2),
                        dtype=np.uint16)  # if not search_kernel_size else np.float32)

    if img.dtype != np.bool:
        img = img.astype(bool)
    elif copy:
        img = img.copy()

    if remove_small_obj_size:
        remove_small_objects(img, remove_small_obj_size,
                             connectivity=2, in_place=True)
    if reconnect_size:
        # remove gaps
        maximum_filter(img, reconnect_size, output=img)
        # reduce contour width to 1
        img = skeletonize(img)

    n_contours = _populateContoursArray(img, contours, minimum_cluster_size)
    contours = contours[:n_contours]

    l = []
    for c in contours:
        ind = np.zeros(shape=len(c), dtype=bool)
        _getValidInd(c, ind)
        # remove all empty spaces:
        l.append(c[ind])
    return l