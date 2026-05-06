def compute_fdm(fixmat, fwhm=2, scale_factor=1):
    """
    Computes a fixation density map for the calling fixmat. 
    
    Creates a map the size of the image fixations were recorded on.  
    Every pixel contains the frequency of fixations
    for this image. The fixation map is smoothed by convolution with a
    Gaussian kernel to approximate the area with highest processing
    (usually 2 deg. visual angle).

    Note: The function does not check whether the fixmat contains
    fixations from different images as it might be desirable to compute
    fdms over fixations from more than one image.

    Parameters:
        fwhm :  float 
            the full width at half maximum of the Gaussian kernel used
            for convolution of the fixation frequency map.

        scale_factor : float
            scale factor for the resulting fdm. Default is 1. Scale_factor
            must be a float specifying the fraction of the current size.
        
    Returns:
        fdm  : numpy.array 
            a numpy.array of size fixmat.image_size containing
            the fixation probability for every location on the image.
    """
    # image category must exist (>-1) and image_size must be non-empty
    assert (len(fixmat.image_size) == 2 and (fixmat.image_size[0] > 0) and
        (fixmat.image_size[1] > 0)), 'The image_size is either 0, or not 2D'
    # check whether fixmat contains fixations
    if fixmat._num_fix == 0 or len(fixmat.x) == 0 or len(fixmat.y) == 0 :
        raise RuntimeError('There are no fixations in the fixmat.')
    assert not scale_factor <= 0, "scale_factor has to be > 0"
    # this specifies left edges of the histogram bins, i.e. fixations between
    # ]0 binedge[0]] are included. --> fixations are ceiled
    e_y = np.arange(0, np.round(scale_factor*fixmat.image_size[0]+1))
    e_x = np.arange(0, np.round(scale_factor*fixmat.image_size[1]+1))
    samples = np.array(list(zip((scale_factor*fixmat.y), (scale_factor*fixmat.x))))
    (hist, _) = np.histogramdd(samples, (e_y, e_x))
    kernel_sigma = fwhm * fixmat.pixels_per_degree * scale_factor
    kernel_sigma = kernel_sigma / (2 * (2 * np.log(2)) ** .5)
    fdm = gaussian_filter(hist, kernel_sigma, order=0, mode='constant')
    return fdm / fdm.sum()