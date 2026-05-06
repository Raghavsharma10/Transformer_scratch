def get_group(self, group_id):
        """Get a single group.

        :param str group_id: group ID

        :returns: group
        :rtype: :class:`marathon.models.group.MarathonGroup`
        """
        response = self._do_request(
            'GET', '/v2/groups/{group_id}'.format(group_id=group_id))
        return self._parse_response(response, MarathonGroup)