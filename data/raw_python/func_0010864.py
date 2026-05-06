def add_entity_info( data_api, struct_inflator):
    """Add the entity info to the structure.
    :param data_api the interface to the decoded data
    :param struct_inflator the interface to put the data into the client object
    """
    for entity in data_api.entity_list:
        struct_inflator.set_entity_info(entity["chainIndexList"],
                                        entity["sequence"],
                                        entity["description"],
                                        entity["type"])