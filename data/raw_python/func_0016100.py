def delete_group(self, group_id, force=False):
        """Stop and destroy a group.

        :param str group_id: group ID
        :param bool force: apply even if a deployment is in progress

        :returns: a dict containing the deleted version
        :rtype: dict
        """
        params = {'force': force}
        response = self._do_request(
            'DELETE', '/v2/groups/{group_id}'.format(group_id=group_id), params=params)
        return response.json()