def url_name_for_action(self, action):
        """
        Returns the reverse name for this action
        """
        return "%s.%s_%s" % (self.module_name.lower(), self.model_name.lower(), action)