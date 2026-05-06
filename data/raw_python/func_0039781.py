def getEdge(self, _id):
        """Retrieves an existing edge from the graph
        @params _id: Edge unique identifier

        @returns The requested Edge or None"""
        try:
            edge = self.neograph.relationships.get(_id)
        except client.NotFoundError:
            return None
        return Edge(edge)