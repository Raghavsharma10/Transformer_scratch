def delete_app(self, app_id, force=False):
        """Stop and destroy an app.

        :param str app_id: application ID
        :param bool force: apply even if a deployment is in progress

        :returns: a dict containing the deployment id and version
        :rtype: dict
        """
        params = {'force': force}
        response = self._do_request(
            'DELETE', '/v2/apps/{app_id}'.format(app_id=app_id), params=params)
        return response.json()