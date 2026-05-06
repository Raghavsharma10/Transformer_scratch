def set_multi(self, mappings, time=0, compress_level=-1):
        """
        Set multiple keys with it's values on server.

        :param mappings: A dict with keys/values
        :type mappings: dict
        :param time: Time in seconds that your key will expire.
        :type time: int
        :param compress_level: How much to compress.
            0 = no compression, 1 = fastest, 9 = slowest but best,
            -1 = default compression level.
        :type compress_level: int
        :return: True in case of success and False in case of failure
        :rtype: bool
        """
        returns = []
        if not mappings:
            return False
        server_mappings = defaultdict(dict)
        for key, value in mappings.items():
            server_key = self._get_server(key)
            server_mappings[server_key].update([(key, value)])
        for server, m in server_mappings.items():
            returns.append(server.set_multi(m, time, compress_level))

        return all(returns)