def redis_key(cls, key):
        """
        Get the key we pass to redis.
        If no namespace is declared, it will use the class name.

        :param key: str     the name of the redis key
        :return: str
        """
        keyspace = cls.keyspace
        tpl = cls.keyspace_template
        key = "%s" % key if keyspace is None else tpl % (keyspace, key)
        return cls.keyparse.encode(key)