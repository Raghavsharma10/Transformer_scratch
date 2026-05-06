def otsu(fpath):
    """
    Returns value of otsu threshold for an image
    """
    img = imread(fpath, as_grey=True)
    thresh = skimage.filter.threshold_otsu(img)

    return thresh