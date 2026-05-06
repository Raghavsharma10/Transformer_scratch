def get_halfs_double(self, vertex_a1, vertex_b1, vertex_a2, vertex_b2):
        """Compute the two parts separated by ``(vertex_a1, vertex_b1)`` and ``(vertex_a2, vertex_b2)``

           Raise a GraphError when ``(vertex_a1, vertex_b1)`` and
           ``(vertex_a2, vertex_b2)`` do not separate the graph in two
           disconnected parts. The edges must be neighbors. If not a GraphError
           is raised. The for vertices must not coincide or a GraphError is
           raised.

           Returns the vertices of the two halfs and the four 'hinge' vertices
           in the correct order, i.e. both ``vertex_a1`` and ``vertex_a2`` are
           in the first half and both ``vertex_b1`` and ``vertex_b2`` are in the
           second half.
        """
        if vertex_a1 not in self.neighbors[vertex_b1]:
            raise GraphError("vertex_a1 must be a neighbor of vertex_b1.")
        if vertex_a2 not in self.neighbors[vertex_b2]:
            raise GraphError("vertex_a2 must be a neighbor of vertex_b2.")

        # find vertex_a_part (and possibly switch vertex_a2, vertex_b2)
        vertex_a_new = set(self.neighbors[vertex_a1])
        vertex_a_new.discard(vertex_b1)
        if vertex_a1 == vertex_b2:
            # we now that we have to swap vertex_a2 and vertex_b2. The algo
            # below will fail otherwise in this 'exotic' case.
            vertex_a2, vertex_b2 = vertex_b2, vertex_a2
            #vertex_a_new.discard(vertex_a2) # in case there is overlap
        if vertex_a1 == vertex_a2:
            vertex_a_new.discard(vertex_b2) # in case there is overlap
        vertex_a_part = set([vertex_a1])

        touched = False # True if (the switched) vertex_a2 has been reached.
        while len(vertex_a_new) > 0:
            pivot = vertex_a_new.pop()
            if pivot == vertex_b1:
                raise GraphError("The graph can not be separated in two halfs. "
                                 "vertex_b1 reached by vertex_a1.")
            vertex_a_part.add(pivot)
            # create a new set that we can modify
            pivot_neighbors = set(self.neighbors[pivot])
            pivot_neighbors -= vertex_a_part
            if pivot == vertex_a2 or pivot == vertex_b2:
                if pivot == vertex_b2:
                    if touched:
                        raise GraphError("The graph can not be separated in "
                                         "two halfs. vertex_b2 reached by "
                                         "vertex_a1.")
                    else:
                        # put them in the correct order
                        vertex_a2, vertex_b2 = vertex_b2, vertex_a2
                pivot_neighbors.discard(vertex_b2)
                touched = True
            vertex_a_new |= pivot_neighbors

        if vertex_a2 not in vertex_a_part:
            raise GraphError("The graph can not be separated in two halfs. "
                             "vertex_a1 can not reach vertex_a2 trough "
                             "vertex_a_part")

        # find vertex_b_part: easy, is just the rest ...
        #vertex_b_part = set(xrange(self.num_vertices)) - vertex_a_part

        # ... but we also want that there is a path in vertex_b_part from
        # vertex_b1 to vertex_b2
        if vertex_b1 == vertex_b2:
            closed = True
        else:
            vertex_b_new = set(self.neighbors[vertex_b1])
            vertex_b_new.discard(vertex_a1)
            vertex_b_part = set([vertex_b1])

            closed = False
            while len(vertex_b_new) > 0:
                pivot = vertex_b_new.pop()
                if pivot == vertex_b2:
                    closed = True
                    break
                pivot_neighbors = set(self.neighbors[pivot])
                pivot_neighbors -= vertex_b_part
                vertex_b_new |= pivot_neighbors
                vertex_b_part.add(pivot)

        if not closed:
            raise GraphError("The graph can not be separated in two halfs. "
                             "vertex_b1 can not reach vertex_b2 trough "
                             "vertex_b_part")

        # finaly compute the real vertex_b_part, the former loop might break
        # early for efficiency.
        vertex_b_part = set(range(self.num_vertices)) - vertex_a_part

        # done!
        return vertex_a_part, vertex_b_part, \
               (vertex_a1, vertex_b1, vertex_a2, vertex_b2)