def change_owners(self, group_id, owner_id):
        """Change the owner of a group.

        .. note:: you must be the owner to change owners

        :param str group_id: the group_id of a group
        :param str owner_id: the ID of the new owner
        :return: the result
        :rtype: :class:`~groupy.api.groups.ChangeOwnersResult`
        """
        url = utils.urljoin(self.url, 'change_owners')
        payload = {
            'requests': [{
                'group_id': group_id,
                'owner_id': owner_id,
            }],
        }
        response = self.session.post(url, json=payload)
        result, = response.data['results']  # should be exactly one
        return ChangeOwnersResult(**result)