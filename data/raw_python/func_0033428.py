def urlretrieve(uri, saveas=None, retries=3, cache_dir=None):
    '''urllib.urlretrieve wrapper'''
    retries = int(retries) if retries else 3
    # FIXME: make random filename (saveas) in cache_dir...
    # cache_dir = cache_dir or CACHE_DIR
    while retries:
        try:
            _path, headers = urllib.urlretrieve(uri, saveas)
        except Exception as e:
            retries -= 1
            logger.warn(
                'Failed getting uri "%s": %s (retry:%s in 1s)' % (
                    uri, e, retries))
            time.sleep(.2)
            continue
        else:
            break
    else:
        raise RuntimeError("Failed to retrieve uri: %s" % uri)
    return _path