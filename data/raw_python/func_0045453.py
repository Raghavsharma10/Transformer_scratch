def get_cache_key(bucket, name, args, kwargs):
    """
    Gets a unique SHA1 cache key for any call to a native tag.
    Use args and kwargs in hash so that the same arguments use the same key
    """
    u = ''.join(map(str, (bucket, name, args, kwargs)))
    return 'native_tags.%s' % sha_constructor(u).hexdigest()