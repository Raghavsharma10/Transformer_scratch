def _get_app_config(self, app_name):
        """
        Returns an app config for the given name, not by label.
        """

        matches = [app_config for app_config in apps.get_app_configs()
                   if app_config.name == app_name]
        if not matches:
            return
        return matches[0]