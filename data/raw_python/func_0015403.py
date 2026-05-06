def flush_all(self, time=0):
        """
        Send a command to server flush|delete all keys.

        :param time: Time to wait until flush in seconds.
        :type time: int
        :return: True in case of success, False in case of failure
        :rtype: bool
        """
        returns = []
        for server in self.servers:
            returns.append(server.flush_all(time))

        return any(returns)