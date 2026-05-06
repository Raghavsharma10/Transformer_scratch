def connect_redis(cls, redis_client, name=None, transaction=False):
        """
        Store the redis connection in our connector instance.

        Do this during your application bootstrapping.

        We call the pipeline method of the redis client.

        The ``redis_client`` can be either a redis or rediscluster client.
        We use the interface, not the actual class.

        That means we can handle either one identically.

        It doesn't matter if you pass in `Redis` or `StrictRedis`.
        the interface for direct redis commands will behave indentically.
        Keyspaces will work with either, but it presents the same interface
        that the Redis class does, not StrictRedis.

        The transaction flag is a boolean value we hold on to and
        pass to the invocation of something equivalent to:

        .. code-block:: python

            redis_client.pipeline(transaction=transation)

        Unlike redis-py, this flag defaults to False.
        You can configure it to always use the MULTI/EXEC flags,
        but I don't see much point.

        If you need transactional support I recommend using a LUA script.

        **RedPipe** is about improving network round-trip efficiency.

        :param redis_client: redis.StrictRedis() or redis.Redis()
        :param name: identifier for the connection, optional
        :param transaction: bool, defaults to False
        :return: None
        """
        connection_pool = redis_client.connection_pool

        if connection_pool.connection_kwargs.get('decode_responses', False):
            raise InvalidPipeline('decode_responses set to True')

        def pipeline_method():
            """
            A closure wrapping the pipeline.

            :return: pipeline object
            """
            return redis_client.pipeline(transaction=transaction)

        # set up the connection.
        cls.connect(pipeline_method=pipeline_method, name=name)