def funcGauss2D(varSizeX, varSizeY, varPosX, varPosY, varSd):

    """ Create 2D Gaussian kernel. Source:
    http://mathworld.wolfram.com/GaussianFunction.html
    """

    varSizeX = int(varSizeX)
    varSizeY = int(varSizeY)

    # aryX and aryY are in reversed order, this seems to be necessary:
    aryY, aryX = sp.mgrid[0:varSizeX,
                          0:varSizeY]

    # The actual creation of the Gaussian array:
    aryGauss = (
        (
            np.power((aryX - varPosX), 2.0) +
            np.power((aryY - varPosY), 2.0)
        ) /
        (2.0 * np.power(varSd, 2.0))
        )
    aryGauss = np.exp(-aryGauss)
    # normalize
    # aryGauss = aryGauss/(2*np.pi*np.power(varSd, 2))

    return aryGauss