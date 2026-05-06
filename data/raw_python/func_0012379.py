def alternate_helper(x, alt_samps, func=None):
    """Helper function for making fgivenx plots of functions with 2 array
    arguments of variable lengths."""
    alt_samps = alt_samps[~np.isnan(alt_samps)]
    arg1 = alt_samps[::2]
    arg2 = alt_samps[1::2]
    return func(x, arg1, arg2)