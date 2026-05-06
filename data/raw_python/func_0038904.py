def config(env=DEFAULT_ENV, default=None, **overrides):
    """Returns configured REDIS dictionary from REDIS_URL."""

    config = {}

    s = os.environ.get(env, default)

    if s:
        config = parse(s)

    overrides = dict([(k.upper(), v) for k, v in overrides.items()])

    config.update(overrides)

    return config