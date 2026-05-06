def _create_node(self, name, parent, children, data):
        """Create new tree node."""
        self._db[name] = {"parent": parent, "children": children, "data": data}