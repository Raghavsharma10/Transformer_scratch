def update_group(self, group_id, group, force=False, minimal=True):
        """Update a group.

        Applies writable settings in `group` to `group_id`
        Note: this method can not be used to rename groups.

        :param str group_id: target group ID
        :param group: group settings
        :type group: :class:`marathon.models.group.MarathonGroup`
        :param bool force: apply even if a deployment is in progress
        :param bool minimal: ignore nulls and empty collections

        :returns: a dict containing the deployment id and version
        :rtype: dict
        """
        # Changes won't take if version is set - blank it for convenience
        group.version = None

        params = {'force': force}
        data = group.to_json(minimal=minimal)

        response = self._do_request(
            'PUT', '/v2/groups/{group_id}'.format(group_id=group_id), data=data, params=params)
        return response.json()