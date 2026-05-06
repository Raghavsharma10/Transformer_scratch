def kldiv_cs_model(prediction, fm):
    """
    Computes Chao-Shen corrected KL-divergence between prediction
    and fdm made from fixations in fm.

    Parameters :
        prediction : np.ndarray
            a fixation density map
        fm : FixMat object
    """
    # compute histogram of fixations needed for ChaoShen corrected kl-div
    # image category must exist (>-1) and image_size must be non-empty
    assert(len(fm.image_size) == 2 and (fm.image_size[0] > 0) and
        (fm.image_size[1] > 0))
    assert(-1 not in fm.category)
    # check whether fixmat contains fixations
    if len(fm.x) ==  0:
        return np.NaN
    (scale_factor, _) = calc_resize_factor(prediction, fm.image_size)
    # this specifies left edges of the histogram bins, i.e. fixations between
    # ]0 binedge[0]] are included. --> fixations are ceiled
    e_y = np.arange(0, np.round(scale_factor*fm.image_size[0]+1))
    e_x = np.arange(0, np.round(scale_factor*fm.image_size[1]+1))
    samples = np.array(list(zip((scale_factor*fm.y), (scale_factor*fm.x))))
    (fdm, _) = np.histogramdd(samples, (e_y, e_x))

    # compute ChaoShen corrected kl-div
    q = np.array(prediction, copy = True)
    q[q == 0] = np.finfo(q.dtype).eps
    q /= np.sum(q)
    (H, pa, la) = chao_shen(fdm)
    q = q[fdm > 0]
    cross_entropy = -np.sum((pa * np.log2(q)) / la)
    return (cross_entropy - H)