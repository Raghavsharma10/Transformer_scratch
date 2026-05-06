def show(self, user, feed, id):
        """
        Show a specific indicator by id

        :param user: feed username
        :param feed: feed name
        :param id: indicator endpoint id [INT]
        :return: dict

        Example:
            ret = Indicator.show('csirtgadgets','port-scanners', '1234')
        """
        uri = '/users/{}/feeds/{}/indicators/{}'.format(user, feed, id)
        return self.client.get(uri)