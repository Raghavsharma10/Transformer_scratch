def list_versions(self, app_id):
        """List the versions of an app.

        :param str app_id: application ID

        :returns: list of versions
        :rtype: list[str]
        """
        response = self._do_request(
            'GET', '/v2/apps/{app_id}/versions'.format(app_id=app_id))
        return [version for version in response.json()['versions']]