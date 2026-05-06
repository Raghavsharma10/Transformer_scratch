def toUIntArray(img, dtype=None, cutNegative=True, cutHigh=True,
                range=None, copy=True):
    '''
    transform a float to an unsigned integer array of a fitting dtype
    adds an offset, to get rid of negative values

    range = (min, max) - scale values between given range

    cutNegative - all values <0 will be set to 0
    cutHigh - set to False to rather scale values to fit
    '''
    mn, mx = None, None
    if range is not None:
        mn, mx = range

    if dtype is None:
        if mx is None:
            mx = np.nanmax(img)
        dtype = np.uint16 if mx > 255 else np.uint8

    dtype = np.dtype(dtype)
    if dtype == img.dtype:
        return img

    # get max px value:
    b = {'uint8': 255,
         'uint16': 65535,
         'uint32': 4294967295,
         'uint64': 18446744073709551615}[dtype.name]

    if copy:
        img = img.copy()

    if range is not None:
        img = np.asfarray(img)
        img -= mn
        # img[img<0]=0
        # print np.nanmin(img), np.nanmax(img), mn, mx, range, b

        img *= b / (mx - mn)
        # print np.nanmin(img), np.nanmax(img), mn, mx, range, b
        img = np.clip(img, 0, b)

    else:
        if cutNegative:
            img[img < 0] = 0
        else:
            # add an offset to all values:
            mn = np.min(img)
            if mn < 0:
                img -= mn  # set minimum to 0

        if cutHigh:
            #ind = img > b
            img[img > b] = b
        else:
            # scale values
            mx = np.nanmax(img)
            img = np.asfarray(img) * (float(b) / mx)

    img = img.astype(dtype)

#     if range is not None and cutHigh:
#         img[ind] = b
    return img