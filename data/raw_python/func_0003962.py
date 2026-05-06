def symmetry_cycles(self):
        """The cycle representations of the graph symmetries"""
        result = set([])
        for symmetry in self.symmetries:
            result.add(symmetry.cycles)
        return result