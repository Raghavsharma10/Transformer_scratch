def nested_dict_to_list(path, dic, exclusion=None):
    """
    Transform nested dict to list
    """
    result = []
    exclusion = ['__self'] if exclusion is None else exclusion

    for key, value in dic.items():

        if not any([exclude in key for exclude in exclusion]):
            if isinstance(value, dict):
                aux = path + key + "/"
                result.extend(nested_dict_to_list(aux, value))
            else:
                if path.endswith("/"):
                    path = path[:-1]

                result.append([path, key, value])

    return result