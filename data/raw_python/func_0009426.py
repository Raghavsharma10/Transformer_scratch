def estimateFromImages(imgs1, imgs2=None, mn_mx=None, nbins=100):
    '''
    estimate the noise level function as stDev over image intensity
    from a set of 2 image groups 
    images at the same position have to show
    the identical setup, so
    imgs1[i] - imgs2[i] = noise
    '''
    if imgs2 is None:
        imgs2 = [None] * len(imgs1)
    else:
        assert len(imgs1) == len(imgs2)

    y_vals = np.empty((len(imgs1), nbins))
    w_vals = np.zeros((len(imgs1), nbins))

    if mn_mx is None:
        print('estimating min and max image value')
        mn = 1e6
        mx = -1e6
        # get min and max image value checking all first images:
        for n, i1 in enumerate(imgs1):
            print('%s/%s' % (n + 1, len(imgs1)))
            i1 = imread(i1)
            mmn, mmx = _getMinMax(i1)
            mn = min(mn, mmn)
            mx = mx = max(mx, mmx)
        print('--> min(%s), max(%s)' % (mn, mx))
    else:
        mn, mx = mn_mx

    x = None
    print('get noise level function')
    for n, (i1, i2) in enumerate(zip(imgs1, imgs2)):
        print('%s/%s' % (n + 1, len(imgs1)))

        i1 = imread(i1)
        if i2 is not None:
            i2 = imread(i2)

        x, y, weights, _ = calcNLF(i1, i2, mn_mx_nbins=(mn, mx, nbins), x=x)
        y_vals[n] = y
        w_vals[n] = weights

    # filter empty places:
    filledPos = np.sum(w_vals, axis=0) != 0
    w_vals = w_vals[:, filledPos]
    y_vals = y_vals[:, filledPos]
    x = x[filledPos]

    y_avg = np.average(np.nan_to_num(y_vals),
                       weights=w_vals,
                       axis=0)
    w_vals = np.sum(w_vals, axis=0)
    w_vals /= w_vals.sum()

    fitParams, fn, i = _evaluate(x, y_avg, w_vals)
    return x, fn, y_avg, y_vals, w_vals, fitParams, i