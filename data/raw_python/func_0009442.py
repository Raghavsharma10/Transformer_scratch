def hog(image, orientations=8, ksize=(5, 5)):
    '''
    returns the Histogram of Oriented Gradients

    :param ksize: convolution kernel size as (y,x) - needs to be odd
    :param orientations: number of orientations in between rad=0 and rad=pi

    similar to http://scikit-image.org/docs/dev/auto_examples/plot_hog.html
    but faster and with less options
    '''
    s0, s1 = image.shape[:2]

    # speed up the process through saving generated kernels:
    try:
        k = hog.kernels[str(ksize) + str(orientations)]
    except KeyError:
        k = _mkConvKernel(ksize, orientations)
        hog.kernels[str(ksize) + str(orientations)] = k

    out = np.empty(shape=(s0, s1, orientations))
    image[np.isnan(image)] = 0

    for i in range(orientations):
        out[:, :, i] = convolve(image, k[i])
    return out