def update_dict(input_dict,key,value):
    '''update_dict will update lists in a dictionary. If the key is not included,
    if will add as new list. If it is, it will append.
    :param input_dict: the dict to update
    :param value: the value to update with
    '''
    if key in input_dict:
        input_dict[key].append(value)
    else:
        input_dict[key] = [value]
    return input_dict