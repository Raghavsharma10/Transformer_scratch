def show(self, user, name, limit=None, lasttime=None):
        """
        Returns a specific Feed from the API

        :param user: feed username
        :param name: feed name
        :param limit: limit the results
        :param lasttime: only show >= lasttime
        :return: dict

        Example:
            ret = feed.show('csirtgadgets', 'port-scanners', limit=5)
        """
        uri = self.client.remote + '/users/{0}/feeds/{1}'.format(user, name)
        return self.client.get(uri, params={'limit': limit, 'lasttime': lasttime})