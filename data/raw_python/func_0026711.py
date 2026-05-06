def draw_hsv(mag, ang, dtype=uint8, fn=None):
    """
    mag must be uint8, uint16, uint32 and 2-D
    ang is in radians (float)
    """
    assert mag.shape == ang.shape
    assert mag.ndim == 2
    maxval = iinfo(dtype).max

    hsv = dstack(((degrees(ang)/2).astype(dtype),  # /2 to keep less than 255
                  ones_like(mag)*maxval,  # maxval must be after in 1-D case
                  cv2.normalize(mag, alpha=0, beta=maxval, norm_type=cv2.NORM_MINMAX)))
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if fn is not None:
        print('writing ' + fn)
        cv2.imwrite(fn, rgb)

    return rgb