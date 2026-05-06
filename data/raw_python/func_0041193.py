def get_dimension_array(array):
    """
    Get dimension of an array getting the number of rows and the max num of
    columns.
    """
    if all(isinstance(el, list) for el in array):
        result = [len(array), len(max([x for x in array], key=len,))]

    # elif array and isinstance(array, list):
    else:
        result = [len(array), 1]

    return result