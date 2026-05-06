def hdel(self, key, *fields):
        """
        Remove the specified fields from the hash stored at `key`.

        Specified fields that do not exist within this hash are ignored.
        If `key` does not exist, it is treated as an empty hash and this
        command returns zero.

        :param key: The key of the hash
        :type key: :class:`str`, :class:`bytes`
        :param fields: iterable of field names to retrieve
        :returns: the number of fields that were removed from the hash,
            not including specified by non-existing fields.
        :rtype: int

        """
        if not fields:
            future = concurrent.TracebackFuture()
            future.set_result(0)
        else:
            future = self._execute([b'HDEL', key] + list(fields))
        return future