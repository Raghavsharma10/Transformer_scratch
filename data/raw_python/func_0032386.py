def connect_redis(redis_client, name=None, transaction=False):
    """
    Connect your redis-py instance to redpipe.

    Example:

    .. code:: python

        redpipe.connect_redis(redis.StrictRedis(), name='users')


    Do this during your application bootstrapping.

    You can also pass a redis-py-cluster instance to this method.

    .. code:: python

        redpipe.connect_redis(rediscluster.StrictRedisCluster(), name='users')


    You are allowed to pass in either the strict or regular instance.

    .. code:: python

        redpipe.connect_redis(redis.StrictRedis(), name='a')
        redpipe.connect_redis(redis.Redis(), name='b')
        redpipe.connect_redis(rediscluster.StrictRedisCluster(...), name='c')
        redpipe.connect_redis(rediscluster.RedisCluster(...), name='d')

    :param redis_client:
    :param name: nickname you want to give to your connection.
    :param transaction:
    :return:
    """
    return ConnectionManager.connect_redis(
        redis_client=redis_client, name=name, transaction=transaction)