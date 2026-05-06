def get_plan(self, nodes=None):
        """
        Retrieve a plan, e.g. a list of fixtures to be loaded sorted on
        dependency.

        :param list nodes: list of nodes to be loaded.
        :return:
        """
        if nodes:
            plan = self.graph.resolve_nodes(nodes)
        else:
            plan = self.graph.resolve_node()

        return plan