def getDepth(pair, connection=None, info=None):
    """Retrieve the depth for the given pair.  Returns a tuple (asks, bids);
    each of these is a list of (price, volume) tuples."""

    if info is not None:
        info.validate_pair(pair)

    if connection is None:
        connection = common.BTCEConnection()

    response = connection.makeJSONRequest("/api/3/depth/%s" % pair)
    if type(response) is not dict:
        raise TypeError("The response is not a dict.")

    depth = response.get(pair)
    if type(depth) is not dict:
        raise TypeError("The pair depth is not a dict.")

    asks = depth.get(u'asks')
    if type(asks) is not list:
        raise TypeError("The response does not contain an asks list.")

    bids = depth.get(u'bids')
    if type(bids) is not list:
        raise TypeError("The response does not contain a bids list.")

    return asks, bids