def new(self, user, name, description=None):
        """
        Creates a new Feed object

        :param user: feed username
        :param name: feed name
        :param description: feed description
        :return: dict
        """
        uri = self.client.remote + '/users/{0}/feeds'.format(user)

        data = {
            'feed': {
                'name': name,
                'description': description
            }
        }

        resp = self.client.post(uri, data)
        return resp