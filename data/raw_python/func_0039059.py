def mine_urls(urls, params=None, callback=None, **kwargs):
    """Concurrently retrieve URLs.

    :param urls: A set of URLs to concurrently retrieve.
    :type urls: iterable

    :param params: (optional) The URL parameters to send with each
                   request.
    :type params: dict

    :param callback: (optional) A callback function to be called on each
                     :py:class:`aiohttp.client.ClientResponse`.

    :param \*\*kwargs: (optional) Arguments that ``get_miner`` takes.
    """
    miner = Miner(**kwargs)
    try:
        miner.loop.add_signal_handler(signal.SIGINT, miner.close)
        miner.loop.run_until_complete(miner.mine_urls(urls, params, callback))
    except RuntimeError:
        pass