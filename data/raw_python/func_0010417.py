def actionnames(self):
        """
        Returns a alphabetical sorted list of tuples with all known
        service- and action-names.
        """
        actions = []
        for service_name in sorted(self.services.keys()):
            action_names = self.services[service_name].actions.keys()
            for action_name in sorted(action_names):
                actions.append((service_name, action_name))
        return actions