def load_app(self, app):
        """
        Tries to load an initial data class for a specified app. If the specified file does not exist,
        an error will be raised. If the class does exist, but it isn't a subclass of `BaseInitialData`
        then None will be returned.
        :param app: The name of the app in which to load the initial data class. This should be the same
            path as defined in settings.INSTALLED_APPS
        :type app: str
        :return: A subclass instance of BaseInitialData or None
        :rtype: BaseInitialData or None
        """
        if self.loaded_apps.get(app):
            return self.loaded_apps.get(app)

        self.loaded_apps[app] = None
        initial_data_class = import_string(self.get_class_path(app))
        if issubclass(initial_data_class, BaseInitialData):
            self.log('Loaded app {0}'.format(app))
            self.loaded_apps[app] = initial_data_class

        return self.loaded_apps[app]