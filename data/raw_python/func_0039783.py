def getInEdges(self, label=None):
        """Gets all the incoming edges of the node. If label
        parameter is provided, it only returns the edges of
        the given label
        @params label: Optional parameter to filter the edges

        @returns A generator function with the incoming edges"""
        if label:
            for edge in self.neoelement.relationships.incoming(types=[label]):
                yield Edge(edge)
        else:
            for edge in self.neoelement.relationships.incoming():
                yield Edge(edge)