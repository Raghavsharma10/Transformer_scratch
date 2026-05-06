def setValues(nxG, nyG, iBeg, iEnd, jBeg, jEnd, data):
    """
    Set setValues
    @param nxG number of global cells in x
    @param nyG number of global cells in y
    @param iBeg global starting index in x
    @param iEnd global ending index in x
    @param jBeg global starting index in y
    @param jEnd global ending index in y
    @param data local array
    """
    nxGHalf = nxG/2.
    nyGHalf = nyG/2.
    nxGQuart = nxGHalf/2.
    nyGQuart = nyGHalf/2.
    for i in range(data.shape[0]):
        iG = iBeg + i
        di = iG - nxG
        for j in range(data.shape[1]):
            jG = jBeg + j
            dj = jG - 0.8*nyG
            data[i, j] = numpy.floor(1.9*numpy.exp(-di**2/nxGHalf**2 - dj**2/nyGHalf**2))