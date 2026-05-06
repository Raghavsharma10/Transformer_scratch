def call_action(self, service_name, action_name, **kwargs):
        """Executes the given action. Raise a KeyError on unkown actions."""
        action = self.services[service_name].actions[action_name]
        return action.execute(**kwargs)