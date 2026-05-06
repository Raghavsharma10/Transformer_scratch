def decr(self, key, value):
        """
        Decrement a key, if it exists, returns it's actual value, if it don't, return 0.
        Minimum value of decrement return is 0.

        :param key: Key's name
        :type key: six.string_types
        :param value: Number to be decremented
        :type value: int
        :return: Actual value of the key on server
        :rtype: int
        """
        returns = []
        for server in self.servers:
            returns.append(server.decr(key, value))

        return returns[0]