def delete(self, key, cas=0):
        """
        Delete a key/value from server. If key does not exist, it returns True.

        :param key: Key's name to be deleted
        :param cas: CAS of the key
        :return: True in case o success and False in case of failure.
        """
        returns = []
        for server in self.servers:
            returns.append(server.delete(key, cas))

        return any(returns)