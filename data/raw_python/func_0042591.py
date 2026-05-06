def getLinesFromLogFile(stream):
    """
    Returns all lines written to the passed in stream
    """
    stream.flush()
    stream.seek(0)
    lines = stream.readlines()
    return lines