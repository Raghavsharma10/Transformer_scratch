def loadSignalFromDelimitedFile(filename, timeColumnIdx=0, dataColumnIdx=1, delimiter=',', skipHeader=0) -> Signal:
    """ reads a delimited file and converts into a Signal
    :param filename: string
    :param timeColumnIdx: 0 indexed column number
    :param dataColumnIdx: 0 indexed column number
    :param delimiter: char
    :return a Signal instance
    """
    data = np.genfromtxt(filename, delimiter=delimiter, skip_header=skipHeader)
    columnCount = data.shape[1]
    if columnCount < timeColumnIdx + 1:
        raise ValueError(
            filename + ' has only ' + columnCount + ' columns,  time values can\'t be at column ' + timeColumnIdx)
    if columnCount < dataColumnIdx + 1:
        raise ValueError(
            filename + ' has only ' + columnCount + ' columns,  data values can\'t be at column ' + dataColumnIdx)

    t = data[:, [timeColumnIdx]]
    samples = data[:, [dataColumnIdx]]
    # calculate fs as the interval between the time samples
    fs = int(round(1 / (np.diff(t, n=1, axis=0).mean()), 0))
    source = Signal(samples.ravel(), fs)
    return source