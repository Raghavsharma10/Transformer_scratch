def convert_data_array(array, filter_func=None, converter_func=None):  # TODO: add copy parameter, otherwise in-place
    '''Filter and convert raw data numpy array (numpy.ndarray).

    Parameters
    ----------
    array : numpy.array
        Raw data array.
    filter_func : function
        Function that takes array and returns true or false for each item in array.
    converter_func : function
        Function that takes array and returns an array or tuple of arrays.

    Returns
    -------
    data_array : numpy.array
        Data numpy array of specified dimension (converter_func) and content (filter_func)
    '''
#     if filter_func != None:
#         if not hasattr(filter_func, '__call__'):
#             raise ValueError('Filter is not callable')
    if filter_func:
        array = array[filter_func(array)]
#     if converter_func != None:
#         if not hasattr(converter_func, '__call__'):
#             raise ValueError('Converter is not callable')
    if converter_func:
        array = converter_func(array)
    return array