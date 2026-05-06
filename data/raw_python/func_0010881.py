def set_group_bond(self, atom_index_one, atom_index_two, bond_order):
        """Add bonds within a group.
        :param atom_index_one: the integer atom index (in the group) of the first partner in the bond
        :param atom_index_two: the integer atom index (in the group) of the second partner in the bond
        :param bond_order: the integer bond order
        """
        self.current_group.bond_atom_list.append(atom_index_one)
        self.current_group.bond_atom_list.append(atom_index_two)
        self.current_group.bond_order_list.append(bond_order)