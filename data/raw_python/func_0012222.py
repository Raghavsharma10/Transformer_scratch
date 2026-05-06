def activate_widget(self, package):
        """
        activate the widget of the given package

        :param str package: name of the package
        """
        cmd, url = DEVICE_URLS["activate_widget"]

        # get widget id for the package
        widget_id = self._get_widget_id(package)
        url = url.format('{}', package, widget_id)

        self.result = self._exec(cmd, url)