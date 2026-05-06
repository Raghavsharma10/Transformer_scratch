def roll_sd(input, window, scale = 1.0, ddof = 0):
    '''Apply a rolling standard deviation function
to an array. This is a simple rolling aggregation of squared
sums.'''
    nobs, i, j, sx, sxx = 0,0,0,0.,0.
    N = len(input)
    sqrt = np.sqrt

    if window > N:
        raise ValueError('Out of bound')

    output = np.ndarray(N-window+1,dtype=input.dtype)

    for val in input[:window]:
        if val == val:
            nobs += 1
            sx += val
            sxx += val*val

    nn = nobs - ddof
    output[j] = NaN if nn<=0 else sqrt(scale * (sxx - sx*sx/nobs) / nn)

    for val in input[window:]:
        prev = input[j]
        if prev == prev:
            sx -= prev
            sxx -= prev*prev
            nobs -= 1

        if val == val:
            nobs += 1
            sx += val
            sxx += val*val

        j += 1
        nn = nobs - ddof
        output[j] = NaN if nn<=0 else sqrt(scale * (sxx - sx*sx/nobs) / nn)

    return output