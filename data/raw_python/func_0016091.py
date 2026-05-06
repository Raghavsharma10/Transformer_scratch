def update_apps(self, apps, force=False, minimal=True):
        """Update multiple apps.

        Applies writable settings in elements of apps either by upgrading existing ones or creating new ones

        :param apps: sequence of application settings
        :param bool force: apply even if a deployment is in progress
        :param bool minimal: ignore nulls and empty collections

        :returns: a dict containing the deployment id and version
        :rtype: dict
        """
        json_repr_apps = []
        for app in apps:
            # Changes won't take if version is set - blank it for convenience
            app.version = None
            json_repr_apps.append(app.json_repr(minimal=minimal))

        params = {'force': force}
        encoder = MarathonMinimalJsonEncoder if minimal else MarathonJsonEncoder
        data = json.dumps(json_repr_apps, cls=encoder, sort_keys=True)

        response = self._do_request(
            'PUT', '/v2/apps', params=params, data=data)
        return response.json()