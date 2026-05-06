def Cube(x, a, b, c, d):
    """Third order polynomial

    Inputs:
    -------
        ``x``: independent variable
        ``a``: coefficient of the third-order term
        ``b``: coefficient of the second-order term
        ``c``: coefficient of the first-order term
        ``d``: additive constant

    Formula:
    --------
        ``a*x^3 + b*x^2 + c*x + d``
    """
    return a * x ** 3 + b * x ** 2 + c * x + d