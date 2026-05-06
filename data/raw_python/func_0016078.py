def get_cache_key(content, **kwargs):
    '''generate cache key'''
    cache_key = ''
    for key in sorted(kwargs.keys()):
        cache_key = '{cache_key}.{key}:{value}'.format(
            cache_key=cache_key,
            key=key,
            value=kwargs[key],
        )

    cache_key = '{content}{cache_key}'.format(
        content=content,
        cache_key=cache_key,
    )

    # fix for non ascii symbols, ensure encoding, python3 hashlib fix
    cache_key = cache_key.encode('utf-8', 'ignore')
    cache_key = md5(cache_key).hexdigest()

    cache_key = '{prefix}.{version}.{language}.{cache_key}'.format(
        prefix=settings.ACTIVE_URL_CACHE_PREFIX,
        version=__version__,
        language=get_language(),
        cache_key=cache_key
    )

    return cache_key