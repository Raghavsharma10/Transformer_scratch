def get_halfs(self, vertex1, vertex2):
        """Split the graph in two halfs by cutting the edge: vertex1-vertex2

           If this is not possible (due to loops connecting both ends), a
           GraphError is raised.

           Returns the vertices in both halfs.
        """
        def grow(origin, other):
            frontier = set(self.neighbors[origin])
            frontier.discard(other)
            result = set([origin])
            while len(frontier) > 0:
                pivot = frontier.pop()
                if pivot == other:
                    raise GraphError("The graph can not be separated in two halfs "
                                     "by disconnecting vertex1 and vertex2.")
                pivot_neighbors = set(self.neighbors[pivot])
                pivot_neighbors -= result
                frontier |= pivot_neighbors
                result.add(pivot)
            return result

        vertex1_part = grow(vertex1, vertex2)
        vertex2_part = grow(vertex2, vertex1)
        return vertex1_part, vertex2_part