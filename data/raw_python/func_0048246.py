def read_stack(stack, plyt=None, laminaprop=None, rho=None, plyts=None, laminaprops=None,
        rhos=None, offset=0., calc_scf=True):
    """Read a laminate stacking sequence data.

    An ``Laminate`` object is returned based on the inputs given.

    Parameters
    ----------
    stack : list
        Angles of the stacking sequence in degrees.
    plyt : float, optional
        When all plies have the same thickness, ``plyt`` can be supplied.
    laminaprop : tuple, optional
        When all plies have the same material properties, ``laminaprop``
        can be supplied.
    rho : float, optional
        Uniform material density to be used for all plies.
    plyts : list, optional
        A list of floats with the thickness of each ply.
    laminaprops : list, optional
        A list of tuples with a laminaprop for each ply.
    rhos : list, optional
        A list of floats with the material density of each ply.
    offset : float, optional
        Offset along the normal axis about the mid-surface, which influences
        the laminate properties.
    calc_scf : bool, optional
        If True, use :method:`.Laminate.calc_scf` to compute shear correction
        factors, otherwise the default value of 5/6 is used

    Notes
    -----
    ``plyt`` or ``plyts`` must be supplied
    ``laminaprop`` or ``laminaprops`` must be supplied

    For orthotropic plies, the ``laminaprop`` should be::

        laminaprop = (E11, E22, nu12, G12, G13, G23)

    For isotropic pliey, the ``laminaprop`` should be::

        laminaprop = (E, E, nu)

    """
    lam = Laminate()
    lam.offset = offset
    lam.stack = stack

    if plyts is None:
        if plyt is None:
            raise ValueError('plyt or plyts must be supplied')
        else:
            plyts = [plyt for i in stack]

    if laminaprops is None:
        if laminaprop is None:
            raise ValueError('laminaprop or laminaprops must be supplied')
        else:
            laminaprops = [laminaprop for i in stack]

    if rhos is None:
        rhos = [rho for i in stack]

    lam.plies = []
    for plyt, laminaprop, theta, rho in zip(plyts, laminaprops, stack, rhos):
        laminaprop = laminaprop
        ply = Lamina()
        ply.theta = float(theta)
        ply.h = plyt
        ply.matobj = read_laminaprop(laminaprop, rho)
        lam.plies.append(ply)

    lam.rebuild()
    lam.calc_constitutive_matrix()
    if calc_scf:
        lam.calc_scf()

    return lam