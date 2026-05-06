def addVertex(self, _id=None):
        """Add param declared for compability with the API. Neo4j
        creates the id automatically
        @params _id: Node unique identifier

        @returns The created Vertex or None"""
        node = self.neograph.nodes.create(_id=_id)
        return Vertex(node)