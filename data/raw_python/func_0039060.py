def mine_items(identifiers, params=None, callback=None, **kwargs):
    """Concurrently retrieve metadata from Archive.org items.

    :param identifiers: A set of Archive.org item identifiers to mine.
    :type identifiers: iterable

    :param params: (optional) The URL parameters to send with each
                   request.
    :type params: dict

    :param callback: (optional) A callback function to be called on each
                     :py:class:`aiohttp.client.ClientResponse`.

    :param \*\*kwargs: (optional) Arguments that ``get_miner`` takes.
    """
    miner = ItemMiner(**kwargs)
    try:
        miner.loop.run_until_complete(miner.mine_items(identifiers, params, callback))
    except RuntimeError:
        miner.loop.close()