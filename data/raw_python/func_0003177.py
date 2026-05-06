def dsr_thurai_2007(D_eq):
    """
    Drop shape relationship function from Thurai2007
    (http://dx.doi.org/10.1175/JTECH2051.1) paper.
    Arguments:
        D_eq: Drop volume-equivalent diameter (mm)

    Returns:
        r: The vertical-to-horizontal drop axis ratio. Note: the Scatterer class
        expects horizontal to vertical, so you should pass 1/dsr_thurai_2007
    """

    if D_eq < 0.7:
        return 1.0
    elif D_eq < 1.5:
        return 1.173 - 0.5165*D_eq + 0.4698*D_eq**2 - 0.1317*D_eq**3 - \
            8.5e-3*D_eq**4
    else:
        return 1.065 - 6.25e-2*D_eq - 3.99e-3*D_eq**2 + 7.66e-4*D_eq**3 - \
            4.095e-5*D_eq**4