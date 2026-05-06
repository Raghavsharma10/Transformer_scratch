def create(self, path, data=None):
        """
        Creates the given node, regardless of whether or not it already
        exists.
        Returns the new node.
        """
        node = self.current[-1]
        path = self._splitpath(path)
        n_items = len(path)
        for n, item in enumerate(path):
            tag, attribs = self._splittag(item)

            # The leaf node is always newly created.
            if n == n_items-1:
                node = node.add(Node(tag, attribs))
                break

            # Parent nodes are only created if they do not exist yet.
            existing = node.get_child(tag, attribs)
            if existing is not None:
                node = existing
            else:
                node = node.add(Node(tag, attribs))
        if data:
            node.text = unquote(data)
        return node