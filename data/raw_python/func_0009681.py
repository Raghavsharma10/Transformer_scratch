def get(self, id):
        """Get a single group by ID.

        :param str id: a group ID
        :return: a group
        :rtype: :class:`~groupy.api.groups.Group`
        """
        url = utils.urljoin(self.url, id)
        response = self.session.get(url)
        return Group(self, **response.data)