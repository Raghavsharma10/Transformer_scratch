def bonded(self, n1, n2, distance):
        """Return the estimated bond type

           Arguments:
            | ``n1``  --  the atom number of the first atom in the bond
            | ``n2``  --  the atom number of the second atom the bond
            | ``distance``  --  the distance between the two atoms

           This method checks whether for the given pair of atom numbers, the
           given distance corresponds to a certain bond length. The best
           matching bond type will be returned. If the distance is a factor
           ``self.bond_tolerance`` larger than a tabulated distance, the
           algorithm will not relate them.
        """
        if distance > self.max_length * self.bond_tolerance:
            return None

        deviation = 0.0
        pair = frozenset([n1, n2])
        result = None
        for bond_type in bond_types:
            bond_length = self.lengths[bond_type].get(pair)
            if (bond_length is not None) and \
               (distance < bond_length * self.bond_tolerance):
                if result is None:
                    result = bond_type
                    deviation = abs(bond_length - distance)
                else:
                    new_deviation = abs(bond_length - distance)
                    if deviation > new_deviation:
                        result = bond_type
                        deviation = new_deviation
        return result