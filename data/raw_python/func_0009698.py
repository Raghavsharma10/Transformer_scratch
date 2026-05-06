def list(self):
        """Return a list of bots.

        :return: all of your bots
        :rtype: :class:`list`
        """
        response = self.session.get(self.url)
        return [Bot(self, **bot) for bot in response.data]