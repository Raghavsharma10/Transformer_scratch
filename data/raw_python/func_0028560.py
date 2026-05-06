def check_redis_connected(client):
    """
    A built-in check to connect to Redis using the given client and see
    if it responds to the ``PING`` command.

    It's automatically added to the list of Dockerflow checks if a
    :class:`~redis.StrictRedis` instances is passed
    to the :class:`~dockerflow.flask.app.Dockerflow` class during
    instantiation, e.g.::

        import redis
        from flask import Flask
        from dockerflow.flask import Dockerflow

        app = Flask(__name__)
        redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

        dockerflow = Dockerflow(app, redis=redis)

    An alternative approach to instantiating a Redis client directly
    would be using the `Flask-Redis <https://github.com/underyx/flask-redis>`_
    Flask extension::

        from flask import Flask
        from flask_redis import FlaskRedis
        from dockerflow.flask import Dockerflow

        app = Flask(__name__)
        app.config['REDIS_URL'] = 'redis://:password@localhost:6379/0'
        redis_store = FlaskRedis(app)

        dockerflow = Dockerflow(app, redis=redis_store)

    """
    import redis
    errors = []

    try:
        result = client.ping()
    except redis.ConnectionError as e:
        msg = 'Could not connect to redis: {!s}'.format(e)
        errors.append(Error(msg, id=health.ERROR_CANNOT_CONNECT_REDIS))
    except redis.RedisError as e:
        errors.append(Error('Redis error: "{!s}"'.format(e),
                            id=health.ERROR_REDIS_EXCEPTION))
    else:
        if not result:
            errors.append(Error('Redis ping failed',
                                id=health.ERROR_REDIS_PING_FAILED))
    return errors