def data_array_from_data_iterable(data_iterable):
    '''Convert data iterable to raw data numpy array.

    Parameters
    ----------
    data_iterable : iterable
        Iterable where each element is a tuple with following content: (raw data, timestamp_start, timestamp_stop, status).

    Returns
    -------
    data_array : numpy.array
        concatenated data array
    '''
    try:
        data_array = np.concatenate([item[0] for item in data_iterable])
    except ValueError:  # length is 0
        data_array = np.empty(0, dtype=np.uint32)
    return data_array