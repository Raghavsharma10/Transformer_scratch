def template_for_action(self, action):
        """
        Returns the template to use for the passed in action
        """
        return "%s/%s_%s.html" % (self.module_name.lower(), self.model_name.lower(), action)