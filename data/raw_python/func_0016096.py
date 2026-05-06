def list_groups(self, **kwargs):
        """List all groups.

        :param kwargs: arbitrary search filters

        :returns: list of groups
        :rtype: list[:class:`marathon.models.group.MarathonGroup`]
        """
        response = self._do_request('GET', '/v2/groups')
        groups = self._parse_response(
            response, MarathonGroup, is_list=True, resource_name='groups')
        for k, v in kwargs.items():
            groups = [o for o in groups if getattr(o, k) == v]
        return groups