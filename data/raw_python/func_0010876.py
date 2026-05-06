def set_group_info(self, group_name, group_number, insertion_code,
                       group_type, atom_count, bond_count, single_letter_code,
                       sequence_index, secondary_structure_type):
        """Set the information for a group
        :param group_name: the name of this group,e.g. LYS
        :param group_number: the residue number of this group
        :param insertion_code: the insertion code for this group
        :param group_type: a string indicating the type of group (as found in the chemcomp dictionary.
        Empty string if none available.
        :param atom_count: the number of atoms in the group
        :param bond_count: the number of unique bonds in the group
        :param single_letter_code: the single letter code of the group
        :param sequence_index: the index of this group in the sequence defined by the enttiy
        :param secondary_structure_type: the type of secondary structure used (types are according to DSSP and
        number to type mappings are defined in the specification)
        """
        # Add the group to the overall list - unless it's the first time round
        if self.current_group is not None:
            self.group_list.append(self.current_group)

        # Add the group level information
        self.group_id_list.append(group_number)
        self.ins_code_list.append(insertion_code)
        self.sequence_index_list.append(sequence_index)
        self.sec_struct_list.append(secondary_structure_type)
        self.current_group = Group()
        self.current_group.group_name = group_name
        self.current_group.group_type = group_type
        self.current_group.single_letter_code = single_letter_code