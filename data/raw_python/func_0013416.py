def limitsSql(startIndex=0, maxResults=0):
    """
    Construct a SQL LIMIT clause
    """
    if startIndex and maxResults:
        return " LIMIT {}, {}".format(startIndex, maxResults)
    elif startIndex:
        raise Exception("startIndex was provided, but maxResults was not")
    elif maxResults:
        return " LIMIT {}".format(maxResults)
    else:
        return ""