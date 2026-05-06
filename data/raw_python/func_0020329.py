def GetNeighboringChannels(channel):
    '''
    Returns all channels on the same module as :py:obj:`channel`.

    '''

    x = divmod(channel - 1, 4)[1]
    return channel + np.array(range(-x, -x + 4), dtype=int)