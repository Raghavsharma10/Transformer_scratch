def _check_graph(self, graph):
        """the atomic numbers must match"""
        if graph.num_vertices != self.size:
            raise TypeError("The number of vertices in the graph does not "
                "match the length of the atomic numbers array.")
        # In practice these are typically the same arrays using the same piece
        # of memory. Just checking to be sure.
        if (self.numbers != graph.numbers).any():
            raise TypeError("The atomic numbers in the graph do not match the "
                "atomic numbers in the molecule.")