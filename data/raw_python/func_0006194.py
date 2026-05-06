def remove_node(self, node):
        """Removes node from circle and rebuild it."""
        try:
            self._nodes.remove(node)
            del self._weights[node]
        except (KeyError, ValueError):
            pass
        self._rebuild_circle()