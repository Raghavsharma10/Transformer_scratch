def select_from_fv_by_seeds(fv, seeds, unique_cls):
    """
    Tool to make simple feature functions take features from feature array by seeds.
    :param fv: ndarray with lineariezed feature. It's shape is MxN, where M is number of image pixels and N is number
    of features
    :param seeds: ndarray with seeds. Does not to be linear.
    :param unique_cls: number of used seeds clases. Like [1, 2]
    :return: fv_selection, seeds_selection - selection from feature vector and selection from seeds
    """
    logger.debug("seeds" + str(seeds))
    # fvlin = fv.reshape(-1, int(fv.size/seeds.size))
    expected_shape = [seeds.size, int(fv.size/seeds.size)]
    if fv.shape[0] != expected_shape[0] or fv.shape[1] != expected_shape[1]:
        raise AssertionError("Wrong shape of input feature vector array fv")
    # sd = seeds.reshape(-1, 1)
    selection = np.in1d(seeds, unique_cls)
    fv_selection = fv[selection]
    seeds_selection = seeds.flatten()[selection]
    # sd = sd[]
    return fv_selection, seeds_selection