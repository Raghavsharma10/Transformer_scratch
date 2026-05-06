def addEdge(self, outVertex, inVertex, label):
        """Creates a new edge
        @params outVertex: Edge origin Vertex
        @params inVertex: Edge target vertex
        @params label: Edge label

        @returns The created Edge object"""
        n1 = outVertex.neoelement
        n2 = inVertex.neoelement
        edge = n1.relationships.create(label, n2)
        return Edge(edge)