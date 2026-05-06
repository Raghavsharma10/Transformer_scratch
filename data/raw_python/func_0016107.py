def get_version(self, app_id, version):
        """Get the configuration of an app at a specific version.

        :param str app_id: application ID
        :param str version: application version

        :return: application configuration
        :rtype: :class:`marathon.models.app.MarathonApp`
        """
        response = self._do_request('GET', '/v2/apps/{app_id}/versions/{version}'
                                    .format(app_id=app_id, version=version))
        return MarathonApp.from_json(response.json())