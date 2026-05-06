def cas(self, key, value, cas, time=0, compress_level=-1):
        """
        Set a value for a key on server if its CAS value matches cas.

        :param key: Key's name
        :type key: six.string_types
        :param value: A value to be stored on server.
        :type value: object
        :param cas: The CAS value previously obtained from a call to get*.
        :type cas: int
        :param time: Time in seconds that your key will expire.
        :type time: int
        :param compress_level: How much to compress.
            0 = no compression, 1 = fastest, 9 = slowest but best,
            -1 = default compression level.
        :type compress_level: int
        :return: True in case of success and False in case of failure
        :rtype: bool
        """
        server = self._get_server(key)
        return server.cas(key, value, cas, time, compress_level)