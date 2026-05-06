def read_lamination_parameters(thickness, laminaprop, rho,
                               xiA1, xiA2, xiA3, xiA4,
                               xiB1, xiB2, xiB3, xiB4,
                               xiD1, xiD2, xiD3, xiD4,
                               xiE1, xiE2, xiE3, xiE4):
    r"""Calculates a laminate based on the lamination parameters.

    The lamination parameters:
    `\xi_{A1} \cdots \xi_{A4}`,  `\xi_{B1} \cdots \xi_{B4}`,
    `\xi_{C1} \cdots \xi_{C4}`,  `\xi_{D1} \cdots \xi_{D4}`,
    `\xi_{E1} \cdots \xi_{E4}`

    are used to calculate the laminate constitutive matrix.

    Parameters
    ----------
    thickness : float
        The total thickness of the laminate
    laminaprop : tuple
        The laminaprop tuple used to define the laminate material.
    rho : float
        Material density.
    xiA1 to xiD4 : float
        The 16 lamination parameters used to define the laminate.

    Returns
    -------
    lam : Laminate
        laminate with the ABD and ABDE matrices already calculated

    """
    lam = Laminate()
    lam.h = thickness
    lam.matobj = read_laminaprop(laminaprop, rho)
    lam.xiA = np.array([1, xiA1, xiA2, xiA3, xiA4], dtype=np.float64)
    lam.xiB = np.array([0, xiB1, xiB2, xiB3, xiB4], dtype=np.float64)
    lam.xiD = np.array([1, xiD1, xiD2, xiD3, xiD4], dtype=np.float64)
    lam.xiE = np.array([1, xiE1, xiE2, xiE3, xiE4], dtype=np.float64)

    lam.calc_ABDE_from_lamination_parameters()
    return lam