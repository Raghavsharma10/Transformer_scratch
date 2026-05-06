def convert_data_iterable(data_iterable, filter_func=None, converter_func=None):  # TODO: add concatenate parameter
    '''Convert raw data in data iterable.

    Parameters
    ----------
    data_iterable : iterable
        Iterable where each element is a tuple with following content: (raw data, timestamp_start, timestamp_stop, status).
    filter_func : function
        Function that takes array and returns true or false for each item in array.
    converter_func : function
        Function that takes array and returns an array or tuple of arrays.

    Returns
    -------
    data_list : list
        Data list of the form [(converted data, timestamp_start, timestamp_stop, status), (...), ...]
    '''
    data_list = []
    for item in data_iterable:
        data_list.append((convert_data_array(item[0], filter_func=filter_func, converter_func=converter_func), item[1], item[2], item[3]))
    return data_list