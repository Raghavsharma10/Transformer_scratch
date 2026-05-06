def create_group(self, group):
        """Create and start a group.

        :param :class:`marathon.models.group.MarathonGroup` group: the group to create

        :returns: success
        :rtype: dict containing the version ID
        """
        data = group.to_json()
        response = self._do_request('POST', '/v2/groups', data=data)
        return response.json()