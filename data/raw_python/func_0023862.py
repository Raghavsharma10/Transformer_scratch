def index(self, user):
        """
        Returns a list of Feeds from the API

        :param user: feed username
        :return: list

        Example:
            ret = feed.index('csirtgadgets')
        """
        uri = self.client.remote + '/users/{0}/feeds'.format(user)
        return self.client.get(uri)