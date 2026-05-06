def ensure_dtraj(dtraj):
    r"""Makes sure that dtraj is a discrete trajectory (array of int)

    """
    if is_int_vector(dtraj):
        return dtraj
    elif is_list_of_int(dtraj):
        return np.array(dtraj, dtype=int)
    else:
        raise TypeError('Argument dtraj is not a discrete trajectory - only list of integers or int-ndarrays are allowed. Check type.')