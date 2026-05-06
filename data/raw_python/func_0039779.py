def getVertex(self, _id):
        """Retrieves an existing vertex from the graph
        @params _id: Node unique identifier

        @returns The requested Vertex or None"""
        try:
            node = self.neograph.nodes.get(_id)
        except client.NotFoundError:
            return None
        return Vertex(node)