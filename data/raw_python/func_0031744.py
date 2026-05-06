def get_user(self, name):
        """Get the user for the given name

        :param name: The username
        :type name: :class:`str`
        :returns: the user instance
        :rtype: :class:`models.User`
        :raises: None
        """
        r = self.kraken_request('GET', 'user/' + name)
        return models.User.wrap_get_user(r)