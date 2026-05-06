def symmetries(self):
        """Graph symmetries (permutations) that map the graph onto itself."""

        symmetry_cycles = set([])
        symmetries = set([])
        for match in GraphSearch(EqualPattern(self))(self):
            match.cycles = match.get_closed_cycles()
            if match.cycles in symmetry_cycles:
                raise RuntimeError("Duplicates in EqualMatch")
            symmetry_cycles.add(match.cycles)
            symmetries.add(match)
        return symmetries