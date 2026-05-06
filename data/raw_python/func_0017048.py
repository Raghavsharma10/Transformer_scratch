def is_following(self, login):
        """Checks if this user is following ``login``.

        :param str login: (required)
        :returns: bool

        """
        url = self.following_urlt.expand(other_user=login)
        return self._boolean(self._get(url), 204, 404)