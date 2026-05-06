def get_family(self, node):
        """
        RETURN ALL ADJACENT NODES
        """
        return set(p if c == node else c for p, c in self.get_edges(node))