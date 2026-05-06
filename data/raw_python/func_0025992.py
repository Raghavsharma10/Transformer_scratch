def loadTriAxisSignalFromFile(filename, timeColumnIdx=0, xIdx=1, yIdx=2, zIdx=3, delimiter=',',
                              skipHeader=0) -> TriAxisSignal:
    """
    A factory method for loading a tri axis measurement from a single file.
    :param filename: the file to load from.
    :param timeColumnIdx: the column containing time data.
    :param xIdx: the column containing x axis data.
    :param yIdx: the column containing y axis data.
    :param zIdx: the column containing z axis data.
    :param delimiter: the delimiter.
    :param skipHeader: how many rows of headers to skip.
    :return: the measurement
    """
    return TriAxisSignal(
        x=loadSignalFromDelimitedFile(filename, timeColumnIdx=timeColumnIdx, dataColumnIdx=xIdx,
                                      delimiter=delimiter, skipHeader=skipHeader),
        y=loadSignalFromDelimitedFile(filename, timeColumnIdx=timeColumnIdx, dataColumnIdx=yIdx,
                                      delimiter=delimiter, skipHeader=skipHeader),
        z=loadSignalFromDelimitedFile(filename, timeColumnIdx=timeColumnIdx, dataColumnIdx=zIdx,
                                      delimiter=delimiter, skipHeader=skipHeader))