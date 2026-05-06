def keys(self):
        """Iterates over the hive's keys.

        Yields WinRegKey namedtuples containing:

            path: path of the key "RootKey\\Key\\..."
            timestamp: date and time of last modification
            values: list of values (("ValueKey", "ValueType", ValueValue), ... )

        """
        for node in self.node_children(self.root()):
            yield from self._visit_registry(node, self._rootkey)