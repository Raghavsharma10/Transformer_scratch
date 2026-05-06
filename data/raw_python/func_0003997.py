def _get_new_ref(self, existing_refs):
        """Get a new reference atom for a row in the ZMatrix

           The reference atoms should obey the following conditions:
             - They must be different
             - They must be neighbours in the bond graph
             - They must have an index lower than the current atom

           If multiple candidate refs can be found, take the heaviest atom
        """
        # ref0 is the atom whose position is defined by the current row in the
        # zmatrix.
        ref0 = existing_refs[0]
        for ref in existing_refs:
            # try to find a neighbor of the ref that can serve as the new ref
            result = None
            for n in sorted(self.graph.neighbors[ref]):
                if self.new_index[n] > self.new_index[ref0]:
                    # index is too high, zmatrix rows can't refer to future
                    # atoms
                    continue
                if n in existing_refs:
                    # ref is already in use
                    continue
                if result is None or self.graph.numbers[n] <= self.graph.numbers[result]:
                    # acceptable ref, prefer heaviest atom
                    result = n
            if result is not None:
                return result
        raise RuntimeError("Could not find new reference.")