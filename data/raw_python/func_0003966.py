def get_subgraph(self, subvertices, normalize=False):
        """Constructs a subgraph of the current graph

           Arguments:
            | ``subvertices`` -- The vertices that should be retained.
            | ``normalize`` -- Whether or not the vertices should renumbered and
                 reduced to the given set of subvertices. When True, also the
                 edges are sorted. It the end, this means that new order of the
                 edges does not depend on the original order, but only on the
                 order of the argument subvertices.
                 This option is False by default. When False, only edges will be
                 discarded, but the retained data remain unchanged. Also the
                 parameter num_vertices is not affected.

           The returned graph will have an attribute ``old_edge_indexes`` that
           relates the positions of the new and the old edges as follows::

             >>> self.edges[result._old_edge_indexes[i]] = result.edges[i]

           In derived classes, the following should be supported::

             >>> self.edge_property[result._old_edge_indexes[i]] = result.edge_property[i]

           When ``normalize==True``, also the vertices are affected and the
           derived classes should make sure that the following works::

             >>> self.vertex_property[result._old_vertex_indexes[i]] = result.vertex_property[i]

           The attribute ``old_vertex_indexes`` is only constructed when
           ``normalize==True``.
        """
        if normalize:
            revorder = dict((j, i) for i, j in enumerate(subvertices))
            new_edges = []
            old_edge_indexes = []
            for counter, (i, j) in enumerate(self.edges):
                new_i = revorder.get(i)
                if new_i is None:
                    continue
                new_j = revorder.get(j)
                if new_j is None:
                    continue
                new_edges.append((new_i, new_j))
                old_edge_indexes.append(counter)
            # sort the edges
            order = list(range(len(new_edges)))
            # argsort in pure python
            order.sort( key=(lambda i: tuple(sorted(new_edges[i]))) )
            new_edges = [new_edges[i] for i in order]
            old_edge_indexes = [old_edge_indexes[i] for i in order]

            result = Graph(new_edges, num_vertices=len(subvertices))
            result._old_vertex_indexes = np.array(subvertices, dtype=int)
            #result.new_vertex_indexes = revorder
            result._old_edge_indexes = np.array(old_edge_indexes, dtype=int)
        else:
            subvertices = set(subvertices)
            old_edge_indexes = np.array([
                i for i, edge in enumerate(self.edges)
                if edge.issubset(subvertices)
            ], dtype=int)
            new_edges = tuple(self.edges[i] for i in old_edge_indexes)
            result = Graph(new_edges, self.num_vertices)
            result._old_edge_indexes = old_edge_indexes
            # no need for old and new vertex_indexes because they remain the
            # same.
        return result