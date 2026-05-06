def update_app(self, app):
        """
        Loads and runs `update_initial_data` of the specified app. Any dependencies contained within the
        initial data class will be run recursively. Dependency cycles are checked for and a cache is built
        for updated apps to prevent updating the same app more than once.
        :param app: The name of the app to update. This should be the same path as defined
            in settings.INSTALLED_APPS
        :type app: str
        """
        # don't update this app if it has already been updated
        if app in self.updated_apps:
            return

        # load the initial data class
        try:
            initial_data_class = self.load_app(app)
        except ImportError as e:
            message = str(e)

            # Check if this error is simply the app not having initial data
            if 'No module named' in message and 'fixtures' in message:
                self.log('No initial data file for {0}'.format(app))
                return
            else:
                # This is an actual import error we should know about
                raise

        self.log('Checking dependencies for {0}'.format(app))

        # get dependency list
        dependencies = self.get_dependency_call_list(app)

        # update initial data of dependencies
        for dependency in dependencies:
            self.update_app(dependency)

        self.log('Updating app {0}'.format(app))

        # Update the initial data of the app and gather any objects returned for deletion. Objects registered for
        # deletion can either be returned from the update_initial_data function or programmatically added with the
        # register_for_deletion function in the BaseInitialData class.
        initial_data_instance = initial_data_class()
        model_objs_registered_for_deletion = initial_data_instance.update_initial_data() or []
        model_objs_registered_for_deletion.extend(initial_data_instance.get_model_objs_registered_for_deletion())

        # Add the objects to be deleted from the app to the global list of objects to be deleted.
        self.model_objs_registered_for_deletion.extend(model_objs_registered_for_deletion)

        # keep track that this app has been updated
        self.updated_apps.add(app)