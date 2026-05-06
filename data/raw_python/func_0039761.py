def average_over_area(q, x, y):
    """Averages a quantity `q` over a rectangular area given a 2D array and
    the x and y vectors for sample locations, using the trapezoidal rule"""
    area = (np.max(x) - np.min(x))*(np.max(y) - np.min(y))
    integral = np.trapz(np.trapz(q, y, axis=0), x)
    return integral/area