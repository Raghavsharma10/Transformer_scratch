def check_redis_connected(app_configs, **kwargs):
    """
    A Django check to connect to the default redis connection
    using ``django_redis.get_redis_connection`` and see if Redis
    responds to a ``PING`` command.
    """
    import redis
    from django_redis import get_redis_connection
    errors = []

    try:
        connection = get_redis_connection('default')
    except redis.ConnectionError as e:
        msg = 'Could not connect to redis: {!s}'.format(e)
        errors.append(checks.Error(msg, id=health.ERROR_CANNOT_CONNECT_REDIS))
    except NotImplementedError as e:
        msg = 'Redis client not available: {!s}'.format(e)
        errors.append(checks.Error(msg, id=health.ERROR_MISSING_REDIS_CLIENT))
    except ImproperlyConfigured as e:
        msg = 'Redis misconfigured: "{!s}"'.format(e)
        errors.append(checks.Error(msg, id=health.ERROR_MISCONFIGURED_REDIS))
    else:
        result = connection.ping()
        if not result:
            msg = 'Redis ping failed'
            errors.append(checks.Error(msg, id=health.ERROR_REDIS_PING_FAILED))
    return errors