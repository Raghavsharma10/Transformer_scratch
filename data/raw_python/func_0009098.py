def update_dict_sum(input_dict,key,increment=None,initial_value=None):
    '''update_dict sum will increment a dictionary key 
    by an increment, and add a value of 0 if it doesn't exist
    :param input_dict: the dict to update
    :param increment: the value to increment by. Default is 1
    :param initial_value: value to start with. Default is 0
    '''
    if increment == None:
        increment = 1

    if initial_value == None:
        initial_value = 0

    if key in input_dict:
        input_dict[key] += increment
    else:
        input_dict[key] = initial_value + increment
    return input_dict