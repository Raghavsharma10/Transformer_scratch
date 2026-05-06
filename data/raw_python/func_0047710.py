def convert_dict_to_datetime(obj_map):
    """converts dictionary representations of datetime back to datetime obj"""
    converted_map = {}
    for key, value in obj_map.items():
        if isinstance(value, dict) and 'tzinfo' in value.keys():
            converted_map[key] = datetime.datetime(**value)
        elif isinstance(value, dict):
            converted_map[key] = convert_dict_to_datetime(value)
        elif isinstance(value, list):
            updated_list = []
            for internal_item in value:
                if isinstance(internal_item, dict):
                    updated_list.append(convert_dict_to_datetime(internal_item))
                else:
                    updated_list.append(internal_item)
            converted_map[key] = updated_list
        else:
            converted_map[key] = value
    return converted_map