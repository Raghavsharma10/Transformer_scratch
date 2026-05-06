def _edges_in_tri_except(self, tri, edge):
        """Return the edges in *tri*, excluding *edge*.
        """
        edges = [(tri[i], tri[(i+1) % 3]) for i in range(3)]
        try:
            edges.remove(tuple(edge))
        except ValueError:
            edges.remove(tuple(edge[::-1]))
        return edges