def limiter(arr):
    """
    Restrict the maximum and minimum values of arr
    """
    dyn_range = 32767.0 / 32767.0
    lim_thresh = 30000.0 / 32767.0
    lim_range = dyn_range - lim_thresh

    new_arr = arr.copy()

    inds = N.where(arr > lim_thresh)[0]

    new_arr[inds] = (new_arr[inds] - lim_thresh) / lim_range
    new_arr[inds] = (N.arctan(new_arr[inds]) * 2.0 / N.pi) *\
        lim_range + lim_thresh

    inds = N.where(arr < -lim_thresh)[0]

    new_arr[inds] = -(new_arr[inds] + lim_thresh) / lim_range
    new_arr[inds] = -(
        N.arctan(new_arr[inds]) * 2.0 / N.pi * lim_range + lim_thresh)

    return new_arr