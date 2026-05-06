def toGray(img):
    '''
    weights see
    https://en.wikipedia.org/wiki/Grayscale#Colorimetric_.28luminance-prese
    http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor
    '''
    return np.average(img, axis=-1, weights=(0.299,  # red
                                             0.587,  # green
                                             0.114)  # blue
                      ).astype(img.dtype)