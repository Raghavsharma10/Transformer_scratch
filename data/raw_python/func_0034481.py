def hmset(self, key, value_dict):
        """
        Sets fields to values as in `value_dict` in the hash stored at `key`.

        Sets the specified fields to their respective values in the hash
        stored at `key`.  This command overwrites any specified fields
        already existing in the hash.  If `key` does not exist, a new  key
        holding a hash is created.

        .. note::

           **Time complexity**: ``O(N)`` where ``N`` is the number of
           fields being set.

        :param key: The key of the hash
        :type key: :class:`str`, :class:`bytes`
        :param value_dict: field to value mapping
        :type value_dict: :class:`dict`
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        if not value_dict:
            future = concurrent.TracebackFuture()
            future.set_result(False)
        else:
            command = [b'HMSET', key]
            command.extend(sum(value_dict.items(), ()))
            future = self._execute(command)
        return future