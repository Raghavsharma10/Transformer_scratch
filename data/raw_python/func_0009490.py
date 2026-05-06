def imread(img, color=None, dtype=None):
    '''
    dtype = 'noUint', uint8, float, 'float', ...
    '''
    COLOR2CV = {'gray': cv2.IMREAD_GRAYSCALE,
                'all': cv2.IMREAD_COLOR,
                None: cv2.IMREAD_ANYCOLOR
                }
    c = COLOR2CV[color]
    if callable(img):
        img = img()
    elif isinstance(img, string_types):
        #         from_file = True
        #         try:
        #             ftype = img[img.find('.'):]
        #             img = READERS[ftype](img)[0]
        #         except KeyError:
        # open with openCV
        # grey - 8 bit
        if dtype in (None, "noUint") or np.dtype(dtype) != np.uint8:
            c |= cv2.IMREAD_ANYDEPTH
        img2 = cv2.imread(img, c)
        if img2 is None:
            raise IOError("image '%s' is not existing" % img)
        img = img2

    elif color == 'gray' and img.ndim == 3:  # multi channel img like rgb
        # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #cannot handle float64
        img = toGray(img)
    # transform array to uint8 array due to openCV restriction
    if dtype is not None:
        if isinstance(img, np.ndarray):
            img = _changeArrayDType(img, dtype, cutHigh=False)

    return img