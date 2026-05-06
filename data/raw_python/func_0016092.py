def rollback_app(self, app_id, version, force=False):
        """Roll an app back to a previous version.

        :param str app_id: application ID
        :param str version: application version
        :param bool force: apply even if a deployment is in progress

        :returns: a dict containing the deployment id and version
        :rtype: dict
        """
        params = {'force': force}
        data = json.dumps({'version': version})
        response = self._do_request(
            'PUT', '/v2/apps/{app_id}'.format(app_id=app_id), params=params, data=data)
        return response.json()