def NumRegressors(npix, pld_order, cross_terms=True):
    '''
    Return the number of regressors for `npix` pixels
    and PLD order `pld_order`.

    :param bool cross_terms: Include pixel cross-terms? Default :py:obj:`True`

    '''

    res = 0
    for k in range(1, pld_order + 1):
        if cross_terms:
            res += comb(npix + k - 1, k)
        else:
            res += npix
    return int(res)