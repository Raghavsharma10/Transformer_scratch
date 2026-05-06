def switch_to_app(self, package):
        """
        activates an app that is specified by package. Selects the first
        app it finds in the app list

        :param package: name of package/app
        :type package: str
        :return: None
        :rtype: None
        """
        log.debug("switching to app '{}'...".format(package))
        cmd, url = DEVICE_URLS["switch_to_app"]
        widget_id = self._get_widget_id(package)

        url = url.format('{}', package, widget_id)

        self.result = self._exec(cmd, url)