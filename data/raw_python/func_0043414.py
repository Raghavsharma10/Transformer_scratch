def clean_cache_key(key):
    """ Replace spaces with '-' and hash if length is greater than 250.
    """
    cache_key = re.sub(r'\s+', '-', key)
    cache_key = smart_str(cache_key)

    if len(cache_key) > 200:
        cache_key = cache_key[:150] + '-' + hashlib.md5(cache_key).hexdigest()

    return cache_key