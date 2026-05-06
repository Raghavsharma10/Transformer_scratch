def flatFieldFromCloseDistance2(images, bgImages=None, calcStd=False,
                                nlf=None, nstd=6):
    '''
    Same as [flatFieldFromCloseDistance]. Differences are:
    ... single-time-effect removal included
    ... returns the standard deviation of the image average [calcStd=True]

    Optional:
    -----------
    calcStd -> set to True to also return the standard deviation
    nlf -> noise level function (callable)
    nstd -> artefact needs to deviate more than [nstd] to be removed
    '''

    if len(images) > 1:

        # start with brightest images
        def fn(img):
            img = imread(img)
            s0, s1 = img.shape[:2]
            # rough approx. of image brightness:
            return -img[::s0 // 10, ::s1 // 10].min()

        images = sorted(images, key=lambda i: fn(i))

        avgBg = getBackground2(bgImages, images[1])

        i0 = imread(images[0], dtype=float) - avgBg
        i1 = imread(images[1], dtype=float) - avgBg

        if nlf is None:
            nlf = oneImageNLF(i0, i1)[0]

        det = SingleTimeEffectDetection(
            (i0, i1), nlf, nStd=nstd, calcVariance=calcStd)

        for i in images[1:]:
            i = imread(i)
            # exclude erroneously darker areas:
            thresh = det.noSTE - nlf(det.noSTE) * nstd
            mask = i > thresh
            # filter STE:
            det.addImage(i, mask)

        ma = det.noSTE

    else:
        ma = imread(images[0], dtype=float) - avgBg

    # fast artifact free maximum:
    mx = median_filter(ma[::10, ::10], 3).max()

    if calcStd:
        return ma / mx, det.mma.var**0.5 / mx

    return ma / mx