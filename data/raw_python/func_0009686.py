def rejoin(self, group_id):
        """Rejoin a former group.

        :param str group_id: the group_id of a group
        :return: the group
        :rtype: :class:`~groupy.api.groups.Group`
        """
        url = utils.urljoin(self.url, 'join')
        payload = {'group_id': group_id}
        response = self.session.post(url, json=payload)
        return Group(self, **response.data)