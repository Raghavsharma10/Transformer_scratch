async def _precache(url, to_type, force=False):
    '''
    Helper function used by precache and precache-named which does the
    actual precaching
    '''
    if force:
        cli.print('%s: force clearing' % url)
        _clear_cache(url)
    cli.print('%s: precaching "%s"' % (url, to_type))
    with autodrain_worker():
        await singletons.workers.async_enqueue_multiconvert(url, to_type)
    result = TypedResource(url, TypeString(to_type))
    cli.print('%s: %s precached at: %s' % (url, to_type, result.cache_path))