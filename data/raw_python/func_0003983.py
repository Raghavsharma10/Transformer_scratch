def get_closed_cycles(self):
        """Return the closed cycles corresponding to this permutation

           The cycle will be normalized to facilitate the elimination of
           duplicates. The following is guaranteed:

           1) If this permutation is represented by disconnected cycles, the
              cycles will be sorted by the lowest index they contain.
           2) Each cycle starts with its lowest index. (unique starting point)
           3) Singletons are discarded. (because they are boring)
        """
        # A) construct all the cycles
        closed_cycles = []
        todo = set(self.forward.keys())
        if todo != set(self.forward.values()):
            raise GraphError("The subject and pattern graph must have the same "
                             "numbering.")
        current_vertex = None
        while len(todo) > 0:
            if current_vertex == None:
                current_vertex = todo.pop()
                current_cycle = []
            else:
                todo.discard(current_vertex)
            current_cycle.append(current_vertex)
            next_vertex = self.get_destination(current_vertex)
            if next_vertex == current_cycle[0]:
                if len(current_cycle) > 1:
                    # bring the lowest element in front
                    pivot = np.argmin(current_cycle)
                    current_cycle = current_cycle[pivot:] + \
                                    current_cycle[:pivot]
                    closed_cycles.append(current_cycle)
                current_vertex = None
            else:
                current_vertex = next_vertex
        # B) normalize the cycle representation
        closed_cycles.sort() # a normal sort is sufficient because only the
                             # first item of each cycle is considered

        # transform the structure into a tuple of tuples
        closed_cycles = tuple(tuple(cycle) for cycle in closed_cycles)
        return closed_cycles