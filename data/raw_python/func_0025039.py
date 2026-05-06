def get_app_guid(self, app_name):
        """
        Returns the GUID for the app instance with
        the given name.
        """
        summary = self.space.get_space_summary()
        for app in summary['apps']:
            if app['name'] == app_name:
                return app['guid']