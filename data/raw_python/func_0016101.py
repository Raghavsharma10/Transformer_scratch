def scale_group(self, group_id, scale_by):
        """Scale a group by a factor.

        :param str group_id: group ID
        :param int scale_by: factor to scale by

        :returns: a dict containing the deployment id and version
        :rtype: dict
        """
        data = {'scaleBy': scale_by}
        response = self._do_request(
            'PUT', '/v2/groups/{group_id}'.format(group_id=group_id), data=json.dumps(data))
        return response.json()