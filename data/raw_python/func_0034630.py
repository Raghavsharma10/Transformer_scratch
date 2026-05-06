def eval(self, script, keys=None, args=None):
        """:meth:`~tredis.RedisClient.eval` and
        :meth:`~tredis.RedisClient.evalsha` are used to evaluate scripts using
        the Lua interpreter built into Redis starting from version 2.6.0.

        The first argument of EVAL is a Lua 5.1 script. The script does not
        need to define a Lua function (and should not). It is just a Lua
        program that will run in the context of the Redis server.

        .. note::

           **Time complexity**: Depends on the script that is executed.

        :param str script: The Lua script to execute
        :param list keys: A list of keys to pass into the script
        :param list args: A list of args to pass into the script
        :return: mixed

        """
        if not keys:
            keys = []
        if not args:
            args = []
        return self._execute([b'EVAL', script, str(len(keys))] + keys + args)