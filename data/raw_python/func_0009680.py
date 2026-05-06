def list_former(self):
        """List all former groups.

        :return: a list of groups
        :rtype: :class:`list`
        """
        url = utils.urljoin(self.url, 'former')
        response = self.session.get(url)
        return [Group(self, **group) for group in response.data]