def get_action_arguments(self, service_name, action_name):
        """
        Returns a list of tuples with all known arguments for the given
        service- and action-name combination. The tuples contain the
        argument-name, direction and data_type.
        """
        return self.services[service_name].actions[action_name].info