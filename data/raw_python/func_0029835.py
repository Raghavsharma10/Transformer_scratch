def getTicker(pair, connection=None, info=None):
    """Retrieve the ticker for the given pair.  Returns a Ticker instance."""

    if info is not None:
        info.validate_pair(pair)

    if connection is None:
        connection = common.BTCEConnection()

    response = connection.makeJSONRequest("/api/3/ticker/%s" % pair)

    if type(response) is not dict:
        raise TypeError("The response is a %r, not a dict." % type(response))
    elif u'error' in response:
        print("There is a error \"%s\" while obtaining ticker %s" % (response['error'], pair))
        ticker = None
    else:
        ticker = Ticker(**response[pair])

    return ticker