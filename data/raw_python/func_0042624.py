def get_children(self, node):
        """Get children."""
        #  if node in self.nodes:
        try:
            index = self.nodes.index(node) + 1
            return [self.nodes[index]]
        except IndexError:
            return []