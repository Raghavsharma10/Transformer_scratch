def _get_argument(self, argument_node):
        """
        Returns a FritzActionArgument instance for the given argument_node.
        """
        argument = FritzActionArgument()
        argument.name = argument_node.find(self.nodename('name')).text
        argument.direction = argument_node.find(self.nodename('direction')).text
        rsv = argument_node.find(self.nodename('relatedStateVariable')).text
        # TODO: track malformed xml-nodes (i.e. misspelled)
        argument.data_type = self.state_variables.get(rsv, None)
        return argument