def local_targets(self):
        """Iterator over the targets defined in this build file."""
        for node in self.node:
            if (node.repo, node.path) == (self.target.repo, self.target.path):
                yield node