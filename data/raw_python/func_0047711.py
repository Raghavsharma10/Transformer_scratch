def convert_ids_to_object_ids(obj_map):
    """converts string representations of _id back to ObjectId obj"""
    converted_map = {}
    for key, value in obj_map.items():
        if key == '_id':
            # hacky, but using alias sends back the whole ID string, like
            #   assessment.Item%3A5758326b4a40452d6eee1fa1%40ODL.MIT.EDU
            # so we have to preserve it
            try:
                converted_map[key] = ObjectId(value)
            except InvalidId:
                converted_map[key] = value
        elif isinstance(value, dict):
            converted_map[key] = convert_ids_to_object_ids(value)
        elif isinstance(value, list):
            new_list = []
            for internal_item in value:
                if isinstance(internal_item, dict):
                    new_list.append(convert_ids_to_object_ids(internal_item))
                else:
                    new_list.append(internal_item)
            converted_map[key] = new_list
        else:
            converted_map[key] = value
    return converted_map