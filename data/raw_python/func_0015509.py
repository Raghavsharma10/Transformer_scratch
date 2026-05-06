def cas(self, key, value, cas, time, compress_level=-1):
        """
        Add a key/value to server ony if it does not exist.

        :param key: Key's name
        :type key: six.string_types
        :param value: A value to be stored on server.
        :type value: object
        :param time: Time in seconds that your key will expire.
        :type time: int
        :param compress_level: How much to compress.
            0 = no compression, 1 = fastest, 9 = slowest but best,
            -1 = default compression level.
        :type compress_level: int
        :return: True if key is added False if key already exists and has a different CAS
        :rtype: bool
        """
        # The protocol CAS value 0 means "no cas".  Calling cas() with that value is
        # probably unintentional.  Don't allow it, since it would overwrite the value
        # without performing CAS at all.
        assert cas != 0, '0 is an invalid CAS value'

        # If we get a cas of None, interpret that as "compare against nonexistant and set",
        # which is simply Add.
        if cas is None:
            return self._set_add_replace('add', key, value, time, compress_level=compress_level)
        else:
            return self._set_add_replace('set', key, value, time, cas=cas, compress_level=compress_level)