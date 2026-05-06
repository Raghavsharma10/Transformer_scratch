def expiring_memoize(obj):
    """Like memoize, but forgets after 10 seconds."""

    cache = obj.cache = {}
    last_access = obj.last_access = defaultdict(int)

    @wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)

        if last_access[key] and last_access[key] + 10 < time():
            if key in cache:
                del cache[key]

        last_access[key] = time()

        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]

    return memoizer