def BorueErukhimovich(q, C, r0, s, t):
    """Borue-Erukhimovich model of microphase separation in polyelectrolytes

    Inputs:
    -------
        ``q``: independent variable
        ``C``: scaling factor
        ``r0``: typical el.stat. screening length
        ``s``: dimensionless charge concentration
        ``t``: dimensionless temperature

    Formula:
    --------
        ``C*(x^2+s)/((x^2+s)(x^2+t)+1)`` where ``x=q*r0``

    Literature:
    -----------
        o Borue and Erukhimovich. Macromolecules (1988) 21 (11) 3240-3249
        o Shibayama and Tanaka. J. Chem. Phys (1995) 102 (23) 9392
        o Moussaid et. al. J. Phys II (France) (1993) 3 (4) 573-594
        o Ermi and Amis. Macromolecules (1997) 30 (22) 6937-6942
    """
    x = q * r0
    return C * (x ** 2 + s) / ((x ** 2 + s) * (x ** 2 + t) + 1)