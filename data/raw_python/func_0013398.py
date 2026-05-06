def timeOneSearch(queryString):
    """
    Returns (search result as JSON string, time elapsed during search)
    """
    startTime = time.clock()
    resultString = backend.runSearchVariants(queryString)
    endTime = time.clock()
    elapsedTime = endTime - startTime
    return resultString, elapsedTime