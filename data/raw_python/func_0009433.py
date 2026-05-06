def _getMinMax(img):
    '''
    Get the a range of image intensities
    that most pixels are in with
    '''
    av = np.mean(img)
    std = np.std(img)
    # define range for segmentation:
    mn = av - 3 * std
    mx = av + 3 * std

    return max(img.min(), mn, 0), min(img.max(), mx)