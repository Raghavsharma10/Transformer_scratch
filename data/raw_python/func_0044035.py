def get_atoms(self, inc_alt_states=False):
        """Returns all atoms in the `Monomer`.

        Parameters
        ----------
        inc_alt_states : bool, optional
            If `True`, will return `Atoms` for alternate states.
        """
        if inc_alt_states:
            return itertools.chain(*[x[1].values() for x in sorted(list(self.states.items()))])
        return self.atoms.values()