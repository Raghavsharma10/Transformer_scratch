def update_app(self, app_id, app, force=False, minimal=True):
        """Update an app.

        Applies writable settings in `app` to `app_id`
        Note: this method can not be used to rename apps.

        :param str app_id: target application ID
        :param app: application settings
        :type app: :class:`marathon.models.app.MarathonApp`
        :param bool force: apply even if a deployment is in progress
        :param bool minimal: ignore nulls and empty collections

        :returns: a dict containing the deployment id and version
        :rtype: dict
        """
        # Changes won't take if version is set - blank it for convenience
        app.version = None

        params = {'force': force}
        data = app.to_json(minimal=minimal)

        response = self._do_request(
            'PUT', '/v2/apps/{app_id}'.format(app_id=app_id), params=params, data=data)
        return response.json()