def has_permissions(self):
        """ Check current user permission set

        Checks the current user permission set against the one being requested
        by the application.
        """
        perms = self.request('/me/permissions')['data'][0].keys()
        return all(k in perms for k in app.config[
            'CANVAS_SCOPE'].split(','))