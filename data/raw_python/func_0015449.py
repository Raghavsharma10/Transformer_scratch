def incr(self, key, value):
        """
        Increment a key, if it exists, returns it's actual value, if it don't, return 0.

        :param key: Key's name
        :type key: six.string_types
        :param value: Number to be incremented
        :type value: int
        :return: Actual value of the key on server
        :rtype: int
        """
        returns = []
        for server in self.servers:
            returns.append(server.incr(key, value))

        return returns[0]