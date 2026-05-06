def iter_breadth_first(self, start=None, do_paths=False, do_duplicates=False):
        """Iterate over the vertices with the breadth first algorithm.

           See http://en.wikipedia.org/wiki/Breadth-first_search for more info.
           If not start vertex is given, the central vertex is taken.

           By default, the distance to the starting vertex is also computed. If
           the path to the starting vertex should be computed instead, set path
           to True.

           When duplicate is True, then vertices that can be reached through
           different  paths of equal length, will be iterated twice. This
           typically only makes sense when path==True.
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
        if do_paths:
            result = (start, 0, (start, ))
        else:
            result = (start, 0)
        yield result
        todo = deque([result])
        while len(todo) > 0:
            if do_paths:
                parent, parent_length, parent_path = todo.popleft()
            else:
                parent, parent_length = todo.popleft()
            current_length = parent_length + 1
            for current in self.neighbors[parent]:
                visited = work[current]
                if visited == -1 or (do_duplicates and visited == current_length):
                    work[current] = current_length
                    if do_paths:
                        current_path = parent_path + (current, )
                        result = (current, current_length, current_path)
                    else:
                        result = (current, current_length)
                    #print "iter_breadth_first", result
                    yield result
                    todo.append(result)