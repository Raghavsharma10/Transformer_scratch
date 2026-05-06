def SavGol(y, win=49):
    '''
    Subtracts a second order Savitsky-Golay filter with window size `win`
    and returns the result. This acts as a high pass filter.

    '''

    if len(y) >= win:
        return y - savgol_filter(y, win, 2) + np.nanmedian(y)
    else:
        return y