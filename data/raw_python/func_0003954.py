def neighbors(self):
        """A dictionary with neighbors

           The dictionary will have the following form:
           ``{vertexX: (vertexY1, vertexY2, ...), ...}``
           This means that vertexX and vertexY1 are connected etc. This also
           implies that the following elements are part of the dictionary:
           ``{vertexY1: (vertexX, ...), vertexY2: (vertexX, ...), ...}``.
        """
        neighbors = dict(
            (vertex, []) for vertex
            in range(self.num_vertices)
        )
        for a, b in self.edges:
            neighbors[a].append(b)
            neighbors[b].append(a)
        # turn lists into frozensets
        neighbors = dict((key, frozenset(val)) for key, val in neighbors.items())
        return neighbors