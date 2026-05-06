def get_oembed_providers():
    """
    Get the list of OEmbed providers.
    """
    global _provider_list, _provider_lock
    if _provider_list is not None:
        return _provider_list

    # Allow only one thread to build the list, or make request to embed.ly.
    _provider_lock.acquire()
    try:
        # And check whether that already succeeded when the lock is granted.
        if _provider_list is None:
            _provider_list = _build_provider_list()
    finally:
        # Always release if there are errors
        _provider_lock.release()

    return _provider_list