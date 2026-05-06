def update_all_apps(self):
        """
        Loops through all app names contained in settings.INSTALLED_APPS and calls `update_app`
        on each one. Handles any object deletions that happened after all apps have been initialized.
        """
        for app in apps.get_app_configs():
            self.update_app(app.name)

        # During update_app, all apps added model objects that were registered for deletion.
        # Delete all objects that were previously managed by the initial data process
        self.handle_deletions()