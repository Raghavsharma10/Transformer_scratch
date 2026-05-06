def centre_of_mass(self):
        """Returns the centre of mass of AMPAL object.

        Notes
        -----
        All atoms are included in calculation, call `centre_of_mass`
        manually if another selection is require.

        Returns
        -------
        centre_of_mass : numpy.array
            3D coordinate for the centre of mass.
        """
        elts = set([x.element for x in self.get_atoms()])
        masses_dict = {e: ELEMENT_DATA[e]['atomic mass'] for e in elts}
        points = [x._vector for x in self.get_atoms()]
        masses = [masses_dict[x.element] for x in self.get_atoms()]
        return centre_of_mass(points=points, masses=masses)