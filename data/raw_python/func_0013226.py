def enter(self, path):
        """
        Enters the given node. Creates it if it does not exist.
        Returns the node.
        """
        self.current.append(self.add(path))
        return self.current[-1]