def preprocess_data(Xs_raw):
    '''Translate the center of mass (COM) of the data to the origin.
    Return the prossed data and the shift of the COM'''
    n = len(Xs_raw)
    Xs_raw_mean = sum(X for X in Xs_raw) / n

    return [X - Xs_raw_mean for X in Xs_raw], Xs_raw_mean