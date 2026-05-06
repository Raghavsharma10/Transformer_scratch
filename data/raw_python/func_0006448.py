def convert_tdc_to_channel(channel):
    ''' Converts TDC words at a given channel to common TDC header (0x4).
    '''
    def f(value):
        filter_func = logical_and(is_tdc_word, is_tdc_from_channel(channel))
        select = filter_func(value)
        value[select] = np.bitwise_and(value[select], 0x0FFFFFFF)
        value[select] = np.bitwise_or(value[select], 0x40000000)
        f.__name__ = "convert_tdc_to_channel_" + str(channel)
        return value
    return f