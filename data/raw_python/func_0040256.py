def missing_nodes(self):
        """The set of targets known as dependencies but not yet defined."""
        missing = set()
        for target_addr, target_attrs in self.graph.node.items():
            if 'target_obj' not in target_attrs:
                missing.add(target_addr)
        return missing