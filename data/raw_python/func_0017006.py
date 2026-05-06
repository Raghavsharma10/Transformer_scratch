def iter_all_users(self, number=-1, etag=None, per_page=None):
        """Iterate over every user in the order they signed up for GitHub.

        :param int number: (optional), number of users to return. Default: -1,
            returns all of them
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :param int per_page: (optional), number of users to list per request
        :returns: generator of :class:`User <github3.users.User>`
        """
        url = self._build_url('users')
        return self._iter(int(number), url, User,
                          params={'per_page': per_page}, etag=etag)