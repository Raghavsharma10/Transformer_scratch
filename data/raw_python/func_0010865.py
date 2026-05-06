def get_bonds(input_group):
    """Utility function to get indices (in pairs) of the bonds."""
    out_list = []
    for i in range(len(input_group.bond_order_list)):
        out_list.append((input_group.bond_atom_list[i * 2], input_group.bond_atom_list[i * 2 + 1],))
    return out_list