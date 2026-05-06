def stisObsCount(input):
    """
    Input: A stis multiextension file
    Output: Number of stis science extensions in input
    """
    count = 0
    toclose = False
    if isinstance(input, str):
        input = fits.open(input)
        toclose = True
    for ext in input:
        if 'extname' in ext.header:
            if (ext.header['extname'].upper() == 'SCI'):
                count += 1
    if toclose:
        input.close()
    return count