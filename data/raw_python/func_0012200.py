def _get_widget_id(self, package_name):
        """
        returns widget_id for given package_name does not care
        about multiple widget ids at the moment, just picks the first

        :param str package_name: package to check for
        :return: id of first widget which belongs to the given package_name
        :rtype: str
        """
        widget_id = ""
        for app in self.get_apps_list():
            if app.package == package_name:
                widget_id = list(app.widgets.keys())[0]

        return widget_id