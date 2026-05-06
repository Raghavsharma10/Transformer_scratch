def Square(x, a, b, c):
    """Second order polynomial

    Inputs:
    -------
        ``x``: independent variable
        ``a``: coefficient of the second-order term
        ``b``: coefficient of the first-order term
        ``c``: additive constant

    Formula:
    --------
        ``a*x^2 + b*x + c``
    """
    return a * x ** 2 + b * x + c