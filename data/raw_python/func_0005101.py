def loggray(x, a, b):
    """Auxiliary function that specifies the logarithmic gray scale.
    a and b are the cutoffs."""
    linval = 10.0 + 990.0 * (x-float(a))/(b-a)
    return (np.log10(linval)-1.0)*0.5 * 255.0