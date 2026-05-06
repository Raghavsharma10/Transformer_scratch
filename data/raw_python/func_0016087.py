def create_app(self, app_id, app, minimal=True):
        """Create and start an app.

        :param str app_id: application ID
        :param :class:`marathon.models.app.MarathonApp` app: the application to create
        :param bool minimal: ignore nulls and empty collections

        :returns: the created app (on success)
        :rtype: :class:`marathon.models.app.MarathonApp` or False
        """
        app.id = app_id
        data = app.to_json(minimal=minimal)
        response = self._do_request('POST', '/v2/apps', data=data)
        if response.status_code == 201:
            return self._parse_response(response, MarathonApp)
        else:
            return False