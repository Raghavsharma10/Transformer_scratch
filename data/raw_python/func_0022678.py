def _tri_from_edge(self, edge):
        """Return the only tri that contains *edge*. If two tris share this
        edge, raise an exception.
        """
        edge = tuple(edge)
        p1 = self._edges_lookup.get(edge, None)
        p2 = self._edges_lookup.get(edge[::-1], None)
        if p1 is None:
            if p2 is None:
                raise RuntimeError("No tris connected to edge %r" % (edge,))
            return edge + (p2,)
        elif p2 is None:
            return edge + (p1,)
        else:
            raise RuntimeError("Two triangles connected to edge %r" % (edge,))