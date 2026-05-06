def getTradeHistory(pair, connection=None, info=None, count=None):
    """Retrieve the trade history for the given pair.  Returns a list of
    Trade instances.  If count is not None, it should be an integer, and
    specifies the number of items from the trade history that will be
    processed and returned."""

    if info is not None:
        info.validate_pair(pair)

    if connection is None:
        connection = common.BTCEConnection()

    response = connection.makeJSONRequest("/api/3/trades/%s" % pair)
    if type(response) is not dict:
        raise TypeError("The response is not a dict.")

    history = response.get(pair)
    if type(history) is not list:
        raise TypeError("The response is a %r, not a list." % type(history))

    result = []

    # Limit the number of items returned if requested.
    if count is not None:
        history = history[:count]

    for h in history:
        h["pair"] = pair
        t = Trade(**h)
        result.append(t)
    return result