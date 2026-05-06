def iter_breadth_first_edges(self, start=None):
        """Iterate over the edges with the breadth first convention.

           We need this for the pattern matching algorithms, but a quick look at
           Wikipedia did not result in a known and named algorithm.

           The edges are yielded one by one, together with the distance of the
           edge from the starting vertex and a flag that indicates whether the
           yielded edge connects two vertices that are at the same distance from
           the starting vertex. If that flag is False, the distance from the
           starting vertex to edge[0] is equal to the distance variable and the
           distance from edge[1] to the starting vertex is equal to distance+1.
           One item has the following format: ((i, j), distance, flag)
        """
        if start is None:
            start = self.central_vertex
        else:
            try:
                start = int(start)
            except ValueError:
                raise TypeError("First argument (start) must be an integer.")
            if start < 0 or start >= self.num_vertices:
                raise ValueError("start must be in the range [0, %i[" %
                                 self.num_vertices)
        from collections import deque
        work = np.zeros(self.num_vertices, int)
        work[:] = -1
        work[start] = 0
        todo = deque([start])
        while len(todo) > 0:
            parent = todo.popleft()
            distance = work[parent]
            for current in self.neighbors[parent]:
                if work[current] == -1:
                    yield (parent, current), distance, False
                    work[current] = distance+1
                    todo.append(current)
                elif work[current] == distance and current > parent:
                    # second equation in elif avoids duplicates
                    yield (parent, current), distance, True
                elif work[current] == distance+1:
                    yield (parent, current), distance, False