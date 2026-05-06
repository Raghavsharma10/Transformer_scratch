def add(self, *tasks):
        """ Interfaces the GraphNode `add` method
        """
        nodes = [x.node for x in tasks]
        self.node.add(*nodes)
        return self