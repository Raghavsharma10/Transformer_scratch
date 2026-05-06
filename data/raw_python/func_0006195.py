def add_node(self, node, weight=1):
        """Adds node to circle and rebuild it."""
        self._nodes.add(node)
        self._weights[node] = weight
        self._rebuild_circle()