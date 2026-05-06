def _intersected_edge(self, edges, cut_edge):
        """ Given a list of *edges*, return the first that is intersected by
        *cut_edge*.
        """
        for edge in edges:
            if self._edges_intersect(edge, cut_edge):
                return edge