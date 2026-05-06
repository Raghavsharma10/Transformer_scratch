def delete_app(self, app_name):
        """
        Delete the given app.

        Will fail intentionally if there are any service
        bindings.  You must delete those first.
        """
        if app_name not in self.space.get_apps():
            logging.warning("App not found so... succeeded?")
            return True

        guid = self.get_app_guid(app_name)
        self.api.delete("/v2/apps/%s" % (guid))