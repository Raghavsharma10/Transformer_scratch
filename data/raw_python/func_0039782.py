def getOutEdges(self, label=None):
        """Gets all the outgoing edges of the node. If label
        parameter is provided, it only returns the edges of
        the given label
        @params label: Optional parameter to filter the edges

        @returns A generator function with the outgoing edges"""
        if label:
            for edge in self.neoelement.relationships.outgoing(types=[label]):
                yield Edge(edge)
        else:
            for edge in self.neoelement.relationships.outgoing():
                yield Edge(edge)