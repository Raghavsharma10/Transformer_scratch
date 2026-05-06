def add_xtalographic_info(data_api, struct_inflator):
    """	 Add the crystallographic data to the structure.
	 :param data_api the interface to the decoded data
	 :param struct_inflator the interface to put the data into the client object"""
    if data_api.unit_cell == None and data_api.space_group is not None:
        struct_inflator.set_xtal_info(data_api.space_group,
                                      constants.UNKNOWN_UNIT_CELL)
    elif data_api.unit_cell is not None and data_api.space_group is None:
        struct_inflator.set_xtal_info(constants.UNKNOWN_SPACE_GROUP,
                                      data_api.unit_cell)
    elif data_api.unit_cell is None and data_api.space_group is None:
        struct_inflator.set_xtal_info(constants.UNKNOWN_SPACE_GROUP,
                                      constants.UNKNOWN_UNIT_CELL)
    else:
        struct_inflator.set_xtal_info(data_api.space_group,
                                      data_api.unit_cell)