def compute_jacobian(ics, coordinates):
    """Construct a Jacobian for the given internal and Cartesian coordinates

       Arguments:
        | ``ics`` -- A list of internal coordinate objects.
        | ``coordinates`` -- A numpy array with Cartesian coordinates,
                             shape=(N,3)

       The return value will be a numpy array with the Jacobian matrix. There
       will be a column for each internal coordinate, and a row for each
       Cartesian coordinate (3*N rows).
    """
    N3 = coordinates.size
    jacobian = numpy.zeros((N3, len(ics)), float)
    for j, ic in enumerate(ics):
        # Let the ic object fill in each column of the Jacobian.
        ic.fill_jacobian_column(jacobian[:,j], coordinates)
    return jacobian