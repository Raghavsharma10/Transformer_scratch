def setupuv(rc):
    """
    Horn Schunck legacy OpenCV function requires we use these old-fashioned cv matrices, not numpy array
    """
    if cv is not None:
        (r, c) = rc
        u = cv.CreateMat(r, c, cv.CV_32FC1)
        v = cv.CreateMat(r, c, cv.CV_32FC1)
        return (u, v)
    else:
        return [None]*2