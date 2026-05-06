def _create_intermediate_nodes(self, name):
        """Create intermediate nodes if hierarchy does not exist."""
        hierarchy = self._split_node_name(name, self.root_name)
        node_tree = [
            self.root_name
            + self._node_separator
            + self._node_separator.join(hierarchy[: num + 1])
            for num in range(len(hierarchy))
        ]
        iobj = [
            (child[: child.rfind(self._node_separator)], child)
            for child in node_tree
            if child not in self._db
        ]
        for parent, child in iobj:
            self._db[child] = {"parent": parent, "children": [], "data": []}
            self._db[parent]["children"] = sorted(
                self._db[parent]["children"] + [child]
            )