def join(self, group_id, share_token):
        """Join a group using a share token.

        :param str group_id: the group_id of a group
        :param str share_token: the share token
        :return: the group
        :rtype: :class:`~groupy.api.groups.Group`
        """
        path = '{}/join/{}'.format(group_id, share_token)
        url = utils.urljoin(self.url, path)
        response = self.session.post(url)
        group = response.data['group']
        return Group(self, **group)