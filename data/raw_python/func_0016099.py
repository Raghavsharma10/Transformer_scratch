def rollback_group(self, group_id, version, force=False):
        """Roll a group back to a previous version.

        :param str group_id: group ID
        :param str version: group version
        :param bool force: apply even if a deployment is in progress

        :returns: a dict containing the deployment id and version
        :rtype: dict
        """
        params = {'force': force}
        response = self._do_request(
            'PUT',
            '/v2/groups/{group_id}/versions/{version}'.format(
                group_id=group_id, version=version),
            params=params)
        return response.json()