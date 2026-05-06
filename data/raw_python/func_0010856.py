def add_group_bonds(data_setters, bond_indices, bond_orders):
    """Add the bonds for this group.
    :param data_setters the class to push the data to
    :param bond_indices the indices of the atoms in the group that
    are bonded (in pairs)
    :param bond_orders the orders of the bonds"""
    for bond_index in range(len(bond_orders)):
        data_setters.set_group_bond(bond_indices[bond_index*2],bond_indices[bond_index*2+1],bond_orders[bond_index])