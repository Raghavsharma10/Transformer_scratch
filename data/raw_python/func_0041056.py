def crossrefs(self):
        """Returns a set of non-local targets referenced by this build file."""
        # TODO: memoize this?
        crefs = set()
        for node in self.node:
            if node.repo != self.target.repo or node.path != self.target.path:
                crefs.add(node)
        return crefs