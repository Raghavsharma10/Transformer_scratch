def set_many(self, nodes):
        """
        Takes nodes dict {uri: content, ...} as argument.
        No return.
        """
        data = self._prepare_nodes(nodes)
        self._set_many(data)