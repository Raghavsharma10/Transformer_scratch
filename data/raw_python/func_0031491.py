def resolve_nodes(self, nodes):
        """
        Resolve a given set of nodes.

        Dependencies of the nodes, even if they are not in the given list will
        also be resolved!

        :param list nodes: List of nodes to be resolved
        :return: A list of resolved nodes
        """
        if not nodes:
            return []
        resolved = []
        for node in nodes:
            if node in resolved:
                continue
            self.resolve_node(node, resolved)
        return resolved