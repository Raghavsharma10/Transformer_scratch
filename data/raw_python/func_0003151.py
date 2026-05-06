def uniform_pdf():
    """Uniform PDF for orientation averaging.

    Returns:
        pdf(x), a function that returns the value of the spherical Jacobian-
        normalized uniform PDF. It is normalized for the interval [0, 180].
    """
    norm_const = 1.0
    def pdf(x):
        return norm_const * np.sin(np.pi/180.0 * x)
    norm_dev = quad(pdf, 0.0, 180.0)[0]
    # ensure that the integral over the distribution equals 1
    norm_const /= norm_dev 
    return pdf