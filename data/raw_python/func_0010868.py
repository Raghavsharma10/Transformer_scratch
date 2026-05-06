def set_atom_info(self, atom_name, serial_number, alternative_location_id,
                      x, y, z, occupancy, temperature_factor, element, charge):
        """Create an atom object an set the information.
        :param atom_name: the atom name, e.g. CA for this atom
        :param serial_number: the serial id of the atom (e.g. 1)
        :param alternative_location_id: the alternative location id for the atom, if present
        :param x: the x coordiante of the atom
        :param y: the y coordinate of the atom
        :param z: the z coordinate of the atom
        :param occupancy: the occupancy of the atom
        :param temperature_factor: the temperature factor of the atom
        :param element: the element of the atom, e.g. C for carbon. According to IUPAC. Calcium  is Ca
        :param charge: the formal atomic charge of the atom
        """
        raise NotImplementedError