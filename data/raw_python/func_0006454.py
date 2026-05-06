def get_trigger_data(value, mode=0):
    '''Returns 31bit trigger counter (mode=0), 31bit timestamp (mode=1), 15bit timestamp and 16bit trigger counter (mode=2)
    '''
    if mode == 2:
        return np.right_shift(np.bitwise_and(value, 0x7FFF0000), 16), np.bitwise_and(value, 0x0000FFFF)
    else:
        return np.bitwise_and(value, 0x7FFFFFFF)