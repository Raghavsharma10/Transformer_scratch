def evalsha(self, sha1, keys=None, args=None):
        """Evaluates a script cached on the server side by its SHA1 digest.
        Scripts are cached on the server side using the
        :meth:`~tredis.RedisClient.script_load` command. The command is
        otherwise identical to :meth:`~tredis.RedisClient.eval`.

        .. note::

           **Time complexity**: Depends on the script that is executed.

        :param str sha1: The sha1 hash of the script to execute
        :param list keys: A list of keys to pass into the script
        :param list args: A list of args to pass into the script
        :return: mixed

        """
        if not keys:
            keys = []
        if not args:
            args = []
        return self._execute([b'EVALSHA', sha1, str(len(keys))] + keys + args)