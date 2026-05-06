def _app_exec(self, package, action, params=None):
        """
        meta method for all interactions with apps

        :param package: name of package/app
        :type package: str
        :param action: the action to be executed
        :type action: str
        :param params: optional parameters for this action
        :type params: dict
        :return: None
        :rtype: None
        """
        # get list of possible commands from app.actions
        allowed_commands = []
        for app in self.get_apps_list():
            if app.package == package:
                allowed_commands = list(app.actions.keys())
                break

        # check if action is in this list
        assert(action in allowed_commands)

        cmd, url = DEVICE_URLS["do_action"]
        # get widget id for the package
        widget_id = self._get_widget_id(package)
        url = url.format('{}', package, widget_id)

        json_data = {"id": action}
        if params is not None:
            json_data["params"] = params

        self.result = self._exec(cmd, url, json_data=json_data)