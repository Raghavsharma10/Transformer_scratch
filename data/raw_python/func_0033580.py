def _clear_cache(url, ts=None):
    '''
    Helper function used by precache and clearcache that clears the cache
    of a given URL and type
    '''
    if ts is None:
        # Clears an entire ForeignResource cache
        res = ForeignResource(url)
        if not os.path.exists(res.cache_path_base):
            cli.printerr('%s is not cached (looked at %s)'
                         % (url, res.cache_path_base))
            return
        cli.print('%s: clearing ALL at %s'
                  % (url, res.cache_path_base))
        res.cache_remove_all()
    else:
        # Clears an entire ForeignResource cache
        res = TypedResource(url, ts)
        if not res.cache_exists():
            cli.printerr('%s is not cached for type %s (looked at %s)'
                         % (url, str(ts), res.cache_path))
            return
        cli.print('%s: clearing "%s" at %s'
                  % (url, str(ts), res.cache_path))
        if os.path.isdir(res.cache_path):
            res.cache_remove_as_dir()
        else:
            res.cache_remove()