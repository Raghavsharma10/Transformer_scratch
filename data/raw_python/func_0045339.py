def complex_data(self):
    '''
    This unpacks the data into a time-series data, of complex values.

    Also, any DC offset from the time-series is removed.

    This is a 1D complex-valued numpy array.
    '''
    cp = np.frombuffer(self.data, dtype='i1').astype(np.float32).view(np.complex64)
    cp = cp - cp.mean()
    return cp