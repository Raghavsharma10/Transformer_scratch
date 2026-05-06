def roll_mean(input, window):
    '''Apply a rolling mean function to an array.
This is a simple rolling aggregation.'''
    nobs, i, j, sum_x = 0,0,0,0.
    N = len(input)

    if window > N:
        raise ValueError('Out of bound')

    output = np.ndarray(N-window+1,dtype=input.dtype)

    for val in input[:window]:
        if val == val:
            nobs += 1
            sum_x += val

    output[j] = NaN if not nobs else sum_x / nobs

    for val in input[window:]:
        prev = input[j]
        if prev == prev:
            sum_x -= prev
            nobs -= 1

        if val == val:
            nobs += 1
            sum_x += val

        j += 1
        output[j] = NaN if not nobs else sum_x / nobs

    return output