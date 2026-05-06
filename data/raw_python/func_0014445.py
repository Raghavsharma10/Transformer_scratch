def shuffle_hosts(self):
        """
        Shuffle hosts so we don't always query the first one.
        Example: using in a webapp with X processes in Y servers, the hosts contacted will be more random.
        The user can also call this function to reshuffle every 'x' seconds or before every request.
        :return:
        """
        if len(self.hosts) > 1:
            random.shuffle(self.hosts)
        return self.hosts