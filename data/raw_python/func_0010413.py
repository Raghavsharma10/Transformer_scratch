def _get_arguments(self, action_node):
        """
        Returns a dictionary of arguments for the given action_node.
        """
        arguments = {}
        argument_nodes = action_node.iterfind(
            r'./ns:argumentList/ns:argument', namespaces={'ns': self.namespace})
        for argument_node in argument_nodes:
            argument = self._get_argument(argument_node)
            arguments[argument.name] = argument
        return arguments