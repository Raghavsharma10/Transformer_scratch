def readmarheader(filename):
    """Read a header from a MarResearch .image file."""
    with open(filename, 'rb') as f:
        intheader = np.fromstring(f.read(10 * 4), np.int32)
        floatheader = np.fromstring(f.read(15 * 4), '<f4')
        strheader = f.read(24)
        f.read(4)
        otherstrings = [f.read(16) for i in range(29)]
    return {'Xsize': intheader[0], 'Ysize': intheader[1], 'MeasTime': intheader[8],
            'BeamPosX': floatheader[7], 'BeamPosY': floatheader[8],
            'Wavelength': floatheader[9], 'Dist': floatheader[10],
            '__Origin__': 'MarResearch .image', 'recordlength': intheader[2],
            'highintensitypixels': intheader[4],
            'highintensityrecords': intheader[5],
            'Date': dateutil.parser.parse(strheader),
            'Detector': 'MARCCD', '__particle__': 'photon'}