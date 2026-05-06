def get_actions(self):
        """Returns a list of FritzAction instances."""
        self._read_state_variables()
        actions = []
        nodes = self.root.iterfind(
            './/ns:action', namespaces={'ns': self.namespace})
        for node in nodes:
            action = FritzAction(self.service.service_type,
                                 self.service.control_url)
            action.name = node.find(self.nodename('name')).text
            action.arguments = self._get_arguments(node)
            actions.append(action)
        return actions