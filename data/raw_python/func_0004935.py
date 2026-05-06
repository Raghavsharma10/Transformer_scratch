def readmar(filename):
    """Read a two-dimensional scattering pattern from a MarResearch .image file.
    """
    hed = header.readmarheader(filename)
    with open(filename, 'rb') as f:
        h = f.read(hed['recordlength'])
        data = np.fromstring(
            f.read(2 * hed['Xsize'] * hed['Ysize']), '<u2').astype(np.float64)
        if hed['highintensitypixels'] > 0:
            raise NotImplementedError(
                'Intensities over 65535 are not yet supported!')
        data = data.reshape(hed['Xsize'], hed['Ysize'])
    return data, hed