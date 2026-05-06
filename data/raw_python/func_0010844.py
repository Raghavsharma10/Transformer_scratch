def permission_for_action(self, action):
        """
        Returns the permission to use for the passed in action
        """
        return "%s.%s_%s" % (self.app_name.lower(), self.model_name.lower(), action)