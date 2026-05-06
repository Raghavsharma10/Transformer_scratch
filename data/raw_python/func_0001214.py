def ensure_caches_alive(max_retries: int = 100,
                        retry_timeout: int = 5,
                        exit_on_failure: bool = True) -> bool:
    """
    Checks every cache backend alias in ``settings.CACHES`` until it becomes available. After ``max_retries``
    attempts to reach any backend are failed it returns ``False``. If ``exit_on_failure`` is set it shuts down with
    ``exit(1)``.

    It sets the ``django-docker-helpers:available-check`` key for every cache backend to ensure
    it's receiving connections. If check is passed the key is deleted.

    :param exit_on_failure: set to ``True`` if there's no sense to continue
    :param int max_retries: a number of attempts to reach cache backend, default is ``100``
    :param int retry_timeout: a timeout in seconds between attempts, default is ``5``
    :return: ``True`` if all backends are available ``False`` if any backend check failed
    """
    for cache_alias in settings.CACHES.keys():
        cache = caches[cache_alias]
        wf('Checking if the cache backed is accessible for the alias `%s`... ' % cache_alias, False)
        for i in range(max_retries):
            try:
                cache.set('django-docker-helpers:available-check', '1')
                assert cache.get('django-docker-helpers:available-check') == '1'
                cache.delete('django-docker-helpers:available-check')
                wf('[+]\n')
                break
            except Exception as e:
                wf(str(e) + '\n')
                sleep(retry_timeout)
        else:
            wf('Tried %s time(s). Shutting down.\n' % max_retries)
            exit_on_failure and exit(1)
            return False
    return True