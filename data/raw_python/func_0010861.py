def add_inter_group_bonds(data_api, struct_inflator):
    """	 Generate inter group bonds.
	 Bond indices are specified within the whole structure and start at 0.
	 :param data_api the interface to the decoded data
	 :param struct_inflator the interface to put the data into the client object"""
    for i in range(len(data_api.bond_order_list)):
        struct_inflator.set_inter_group_bond(data_api.bond_atom_list[i * 2],
                                             data_api.bond_atom_list[i * 2 + 1],
                                             data_api.bond_order_list[i])