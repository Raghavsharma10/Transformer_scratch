def canonical_order(self):
        """The vertices in a canonical or normalized order.

           This routine will return a list of vertices in an order that does not
           depend on the initial order, but only depends on the connectivity and
           the return values of the function self.get_vertex_string.

           Only the vertices that are involved in edges will be included. The
           result can be given as first argument to self.get_subgraph, with
           reduce=True as second argument. This will return a complete canonical
           graph.

           The routine is designed not to use symmetry relations that are
           obtained with the GraphSearch routine. We also tried to create an
           ordering that feels like natural, i.e. starting in the center and
           pushing vertices with few equivalents to the front. If necessary, the
           nature of the vertices and  their bonds to atoms closer to the center
           will also play a role, but only as a last resort.
        """
        # A) find an appropriate starting vertex.
        # Here we take a central vertex that has a minimal number of symmetrical
        # equivalents, 'the highest atom number', and the highest fingerprint.
        # Note that the symmetrical equivalents are computed from the vertex
        # fingerprints, i.e. without the GraphSearch.
        starting_vertex = max(
            (
                -len(self.equivalent_vertices[vertex]),
                self.get_vertex_string(vertex),
                self.vertex_fingerprints[vertex].tobytes(),
                vertex
            ) for vertex in self.central_vertices
        )[-1]

        # B) sort all vertices based on
        #      1) distance from central vertex
        #      2) number of equivalent vertices
        #      3) vertex string, (higher atom numbers come first)
        #      4) fingerprint
        #      5) vertex index
        # The last field is only included to collect the result of the sort.
        # The fingerprint on itself would be sufficient, but the three first are
        # there to have a naturally appealing result.
        l = [
            [
                -distance,
                -len(self.equivalent_vertices[vertex]),
                self.get_vertex_string(vertex),
                self.vertex_fingerprints[vertex].tobytes(),
                vertex
            ] for vertex, distance in self.iter_breadth_first(starting_vertex)
            if len(self.neighbors[vertex]) > 0
        ]
        l.sort(reverse=True)

        # C) The order of some vertices is still not completely set. e.g.
        # consider the case of allene. The four hydrogen atoms are equivalent,
        # but one can have two different orders: make geminiles consecutive or
        # don't. It is more trikcy than one would think at first sight. In the
        # case of allene, geminility could easily solve the problem. Consider a
        # big flat rotationally symmetric molecule (order 2). The first five
        # shells are order 4 and one would just give a random order to four
        # segemnts in the first shell. Only when one reaches the outer part that
        # has order two, it turns out that the arbitrary choices in the inner
        # shell play a role. So it does not help to look at relations with
        # vertices at inner or current shells only. One has to consider the
        # whole picture. (unit testing reveals troubles like these)

        # I need some sleep now. The code below checks for potential fuzz and
        # will raise an error if the ordering is not fully determined yet. One
        # day, I'll need this code more than I do now, and I'll fix things up.
        # I know how to do this, but I don't care enough right now.
        # -- Toon
        for i in range(1, len(l)):
            if l[i][:-1] == l[i-1][:-1]:
                raise NotImplementedError

        # D) Return only the vertex indexes.
        return [record[-1] for record in l]