def add_atom_data(data_api, data_setters, atom_names, element_names, atom_charges, group_atom_ind):
    """Add the atomic data to the DataTransferInterface.
    :param data_api the data api from where to get the data
    :param data_setters the class to push the data to
    :param atom_nams the list of atom names for the group
    :param element_names the list of element names for this group
    :param atom_charges the list formal atomic charges for this group
    :param group_atom_ind the index of this atom in the group"""
    atom_name = atom_names[group_atom_ind]
    element = element_names[group_atom_ind]
    charge = atom_charges[group_atom_ind]
    alternative_location_id = data_api.alt_loc_list[data_api.atom_counter]
    serial_number = data_api.atom_id_list[data_api.atom_counter]
    x = data_api.x_coord_list[data_api.atom_counter]
    y = data_api.y_coord_list[data_api.atom_counter]
    z = data_api.z_coord_list[data_api.atom_counter]
    occupancy = data_api.occupancy_list[data_api.atom_counter]
    temperature_factor = data_api.b_factor_list[data_api.atom_counter]
    data_setters.set_atom_info(atom_name, serial_number, alternative_location_id,
                               x, y, z, occupancy, temperature_factor, element, charge)