def set_inter_group_bond(self, atom_index_one, atom_index_two, bond_order):
        """Add bonds between groups.
        :param atom_index_one: the integer atom index (in the structure) of the first partner in the bond
        :param atom_index_two: the integer atom index (in the structure) of the second partner in the bond
        :param bond_order the bond order
        """
        self.bond_atom_list.append(atom_index_one)
        self.bond_atom_list.append(atom_index_two)
        self.bond_order_list.append(bond_order)