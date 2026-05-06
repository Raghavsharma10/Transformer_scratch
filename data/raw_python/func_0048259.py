def depends(self, *tasks):
        """ Interfaces the GraphNode `depends` method """
        nodes = [x.node for x in tasks]
        self.node.depends(*nodes)
        return self