def _createAbsMagEstimationDict():
    """ loads magnitude_estimation.dat which is from
    http://xoomer.virgilio.it/hrtrace/Sk.htm on 24/01/2014 and based on
    Schmid-Kaler (1982)

    creates a dict in the form [Classletter][ClassNumber][List of values for
    each L Class]
    """
    magnitude_estimation_filepath = resource_filename(
        __name__, 'data/magnitude_estimation.dat')
    raw_table = np.loadtxt(magnitude_estimation_filepath, '|S5')

    absMagDict = {
        'O': {},
        'B': {},
        'A': {},
        'F': {},
        'G': {},
        'K': {},
        'M': {}}
    for row in raw_table:
        if sys.hexversion >= 0x03000000:
            # otherwise we get byte ints or b' caused by 2to3
            starClass = row[0].decode("utf-8")
            absMagDict[starClass[0]][int(starClass[1])] = [
                float(x) for x in row[1:]]
        else:
            # dict of spectral type = {abs mag for each luminosity class}
            absMagDict[row[0][0]][int(row[0][1])] = [float(x) for x in row[1:]]

    # manually typed from table headers - used to match columns with the L
    # class (header)
    LClassRef = {
        'V': 0,
        'IV': 1,
        'III': 2,
        'II': 3,
        'Ib': 4,
        'Iab': 5,
        'Ia': 6,
        'Ia0': 7}

    return absMagDict, LClassRef