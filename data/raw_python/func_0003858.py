def url(self):
        """
        Returns the whole URL from the base to this node.
        """
        path = None
        nodes = self.parents()
        while not nodes.empty():
            path = urljoin(path, nodes.get().path())
        return path