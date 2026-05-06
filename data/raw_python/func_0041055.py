def validate_internal_deps(self):
        """Freak out if there are missing local references."""
        for node in self.node:
            if ('target_obj' not in self.node[node]
                    and node not in self.crossrefs):
                raise error.BrokenGraph('Missing target: %s referenced from %s'
                                        ' but not defined there.' %
                                        (node, self.name))