def get_dependency_call_list(self, app, call_list=None):
        """
        Recursively detects any dependency cycles based on the specific app. A running list of
        app calls is kept and passed to each function call. If a circular dependency is detected
        an `InitialDataCircularDependency` exception will be raised. If a dependency does not exist,
        an `InitialDataMissingApp` exception will be raised.
        :param app: The name of the app in which to detect cycles. This should be the same path as defined
            in settings.INSTALLED_APPS
        :type app: str
        :param call_list: A running list of which apps will be updated in this branch of dependencies
        :type call_list: list
        """
        # start the call_list with the current app if one wasn't passed in recursively
        call_list = call_list or [app]

        # load the initial data class for the app
        try:
            initial_data_class = self.load_app(app)
        except ImportError:
            raise InitialDataMissingApp(dep=app)

        dependencies = initial_data_class.dependencies
        # loop through each dependency and check recursively for cycles
        for dep in dependencies:
            if dep in call_list:
                raise InitialDataCircularDependency(dep=dep, call_list=call_list)
            else:
                self.get_dependency_call_list(dep, call_list + [dep])
        call_list.extend(dependencies)

        return call_list[1:]