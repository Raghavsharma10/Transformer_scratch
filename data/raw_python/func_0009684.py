def destroy(self, id):
        """Destroy a group.

        :param str id: a group ID
        :return: ``True`` if successful
        :rtype: bool
        """
        path = '{}/destroy'.format(id)
        url = utils.urljoin(self.url, path)
        response = self.session.post(url)
        return response.ok