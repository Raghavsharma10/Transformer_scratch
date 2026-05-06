def add(self, path, data=None, replace=False):
        """
        Creates the given node if it does not exist.
        Returns the (new or existing) node.
        """
        node = self.current[-1]
        for item in self._splitpath(path):
            tag, attribs = self._splittag(item)
            next_node = node.get_child(tag, attribs)
            if next_node is not None:
                node = next_node
            else:
                node = node.add(Node(tag, attribs))
        if replace:
            node.text = ''
        if data:
            if node.text is None:
                node.text = unquote(data)
            else:
                node.text += unquote(data)
        return node