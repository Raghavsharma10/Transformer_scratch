def then(self, *tasks):
        """ Interfaces the GraphNode `then` method
        """
        nodes = [x.node for x in tasks]
        self.node.then(*nodes)
        return self