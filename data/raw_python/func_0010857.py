def add_group(data_api, data_setters, group_index):
    """Add the data for a whole group.
    :param data_api the data api from where to get the data
    :param data_setters the class to push the data to
    :param group_index the index for this group"""
    group_type_ind = data_api.group_type_list[group_index]
    atom_count = len(data_api.group_list[group_type_ind]["atomNameList"])
    insertion_code = data_api.ins_code_list[group_index]
    data_setters.set_group_info(data_api.group_list[group_type_ind]["groupName"],
                                data_api.group_id_list[group_index], insertion_code,
                                data_api.group_list[group_type_ind]["chemCompType"],
                                atom_count, data_api.num_bonds,
                                data_api.group_list[group_type_ind]["singleLetterCode"],
                                data_api.sequence_index_list[group_index],
                                data_api.sec_struct_list[group_index])
    for group_atom_ind in range(atom_count):
        add_atom_data(data_api, data_setters,
                      data_api.group_list[group_type_ind]["atomNameList"],
                      data_api.group_list[group_type_ind]["elementList"],
                      data_api.group_list[group_type_ind]["formalChargeList"],
                      group_atom_ind)
        data_api.atom_counter +=1
    add_group_bonds(data_setters,
                    data_api.group_list[group_type_ind]["bondAtomList"],
                    data_api.group_list[group_type_ind]["bondOrderList"])
    return atom_count