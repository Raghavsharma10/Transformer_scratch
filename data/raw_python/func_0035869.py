def _createMagConversionDict():
    """ loads magnitude_conversion.dat which is table A% 1995ApJS..101..117K
    """
    magnitude_conversion_filepath = resource_stream(__name__, 'data/magnitude_conversion.dat')
    raw_table = np.loadtxt(magnitude_conversion_filepath, '|S5')

    magDict = {}
    for row in raw_table:
        if sys.hexversion >= 0x03000000:
            starClass = row[1].decode("utf-8")  # otherwise we get byte ints or b' caused by 2to3
            tableData = [x.decode("utf-8") for x in row[3:]]
        else:
            starClass = row[1]
            tableData = row[3:]
        magDict[starClass] = tableData

    return magDict