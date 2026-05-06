def replace(self, key, value, time=0, compress_level=-1):
        """
        Replace a key/value to server ony if it does exist.

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
        :return: True if key is replace False if key does not exists
        :rtype: bool
        """
        server = self._get_server(key)
        return server.replace(key, value, time, compress_level)