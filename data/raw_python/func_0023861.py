def delete(self, user, name):
        """
        Removes a feed

        :param user: feed username
        :param name: feed name
        :return: true/false
        """

        uri = self.client.remote + '/users/{}/feeds/{}'.format(user, name)

        resp = self.client.session.delete(uri)
        return resp.status_code