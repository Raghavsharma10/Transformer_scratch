def hmget(self, key, *fields):
        """
        Returns the values associated with the specified `fields` in a hash.

        For every ``field`` that does not exist in the hash, :data:`None`
        is returned.  Because a non-existing keys are treated as empty
        hashes, calling :meth:`hmget` against a non-existing key will
        return a list of :data:`None` values.

        .. note::

           *Time complexity*: ``O(N)`` where ``N`` is the number of fields
           being requested.

        :param key: The key of the hash
        :type key: :class:`str`, :class:`bytes`
        :param fields: iterable of field names to retrieve
        :returns: a :class:`dict` of field name to value mappings for
            each of the requested fields
        :rtype: dict

        """

        def format_response(val_array):
            return dict(zip(fields, val_array))

        command = [b'HMGET', key]
        command.extend(fields)
        return self._execute(command, format_callback=format_response)