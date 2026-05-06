def hincrbyfloat(self, key, field, increment):
        """
        Increments the number stored at `field` in the hash stored at `key`.

        If the increment value is negative, the result is to have the hash
        field **decremented** instead of incremented.  If the field does not
        exist, it is set to ``0`` before performing the operation.  An error
        is returned if one of the following conditions occur:

        - the field contains a value of the wrong type (not a string)
        - the current field content or the specified increment are not
          parseable as a double precision floating point number

        .. note::

           *Time complexity*: ``O(1)``

        :param key: The key of the hash
        :type key: :class:`str`, :class:`bytes`
        :param field: name of the field to increment
        :type key: :class:`str`, :class:`bytes`
        :param increment: amount to increment by
        :type increment: float

        :returns: the value at `field` after the increment occurs
        :rtype: float

        """
        return self._execute(
            [b'HINCRBYFLOAT', key, field, increment], format_callback=float)