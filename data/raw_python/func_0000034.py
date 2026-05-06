def return_fv_by_seeds(fv, seeds=None, unique_cls=None):
    """
    Return features selected by seeds and unique_cls or selection from features and corresponding seed classes.

    :param fv: ndarray with lineariezed feature. It's shape is MxN, where M is number of image pixels and N is number
    of features
    :param seeds: ndarray with seeds. Does not to be linear.
    :param unique_cls: number of used seeds clases. Like [1, 2]
    :return: fv, sd - selection from feature vector and selection from seeds or just fv for whole image
    """
    if seeds is not None:
        if unique_cls is not None:
            return select_from_fv_by_seeds(fv, seeds, unique_cls)
        else:
            raise AssertionError("Input unique_cls has to be not None if seeds is not None.")
    else:
        return fv