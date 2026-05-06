def uri_exists_wait(uri, timeout=300, interval=5, storage_args={}):
    """
    Block / waits until URI exists.

    :param str uri: URI to check existence
    :param float timeout: Number of seconds before timing out
    :param float interval: Calls :func:`uri_exists` every ``interval`` seconds
    :param dict storage_args: Keyword arguments to pass to the underlying storage object
    :returns: ``True`` if URI exists
    :rtype: bool
    """

    uri_obj = get_uri_obj(uri, storage_args)
    start_time = time.time()
    while time.time() - start_time < timeout:
        if uri_obj.exists(): return True
        time.sleep(interval)
    #end while

    if uri_exists(uri): return True

    return False