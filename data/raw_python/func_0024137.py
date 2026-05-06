def check_valence(self):
        """
        check valences of all atoms

        :return: list of invalid atoms
        """
        return [x for x, atom in self.atoms() if not atom.check_valence(self.environment(x))]