def benchmarkOneQuery(request, repeatLimit=3, pageLimit=3):
    """
    Repeat the query several times; perhaps don't go through *all* the
    pages.  Returns minimum time to run backend.searchVariants() to execute
    the query (as far as pageLimit allows), *not* including JSON
    processing to prepare queries or parse responses.
    """
    times = []
    queryString = protocol.toJson(request)
    for i in range(0, repeatLimit):
        resultString, elapsedTime = timeOneSearch(queryString)
        accruedTime = elapsedTime
        pageCount = 1
        token = extractNextPageToken(resultString)
        # Iterate to go beyond the first page of results.
        while token is not None and pageCount < pageLimit:
            pageRequest = request
            pageRequest.page_token = token
            pageRequestString = protocol.toJson(pageRequest)
            resultString, elapsedTime = timeOneSearch(pageRequestString)
            accruedTime += elapsedTime
            pageCount = pageCount + 1
            token = extractNextPageToken(resultString)
        times.append(accruedTime)

    # TODO: more sophisticated statistics. Sometimes we want min(),
    # sometimes mean = sum() / len(), sometimes other measures,
    # perhaps exclude outliers...

    # If we compute average we should throw out at least the first one.
    # return sum(times[2:])/len(times[2:])
    return min(times)