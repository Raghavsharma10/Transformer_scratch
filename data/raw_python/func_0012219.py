def set_apps_list(self):
        """
        gets installed apps and puts them into the available_apps list
        """
        log.debug("getting apps and setting them in the internal app list...")

        cmd, url = DEVICE_URLS["get_apps_list"]
        result = self._exec(cmd, url)

        self.available_apps = [
            AppModel(result[app])
            for app in result
        ]