def is_data_from_channel(channel=4):  # function factory
    '''Selecting FE data from given channel.

    Parameters
    ----------
    channel : int
        Channel number (4 is default channel on Single Chip Card).

    Returns
    -------
    Function.

    Usage:
    1 Selecting FE data from channel 4 (combine with is_fe_word):
        filter_fe_data_from_channel_4 = logical_and(is_fe_word, is_data_from_channel(4))
        fe_data_from_channel_4 = data_array[filter_fe_data_from_channel_4(data_array)]
    2 Sleceting data from channel 4:
        filter_data_from_channel_4 = is_data_from_channel(4)
        data_from_channel_4 = data_array[filter_data_from_channel_4(fe_data_array)]
    3 Sleceting data from channel 4:
        data_from_channel_4 = is_data_from_channel(4)(fe_raw_data)

    Other usage:
    f_ch4 = functoools.partial(is_data_from_channel, channel=4)
    l_ch4 = lambda x: is_data_from_channel(x, channel=4)
    '''
    if channel >= 0 and channel < 16:
        def f(value):
            return np.equal(np.right_shift(np.bitwise_and(value, 0x0F000000), 24), channel)
        f.__name__ = "is_data_from_channel_" + str(channel)  # or use inspect module: inspect.stack()[0][3]
        return f
    else:
        raise ValueError('Invalid channel number')