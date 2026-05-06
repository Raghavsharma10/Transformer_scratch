def ckm_standard(t12, t13, t23, delta):
    r"""CKM matrix in the standard parametrization and standard phase
    convention.

    Parameters
    ----------

    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `delta`: CKM phase $\delta=\gamma$ in radians
    """
    c12 = cos(t12)
    c13 = cos(t13)
    c23 = cos(t23)
    s12 = sin(t12)
    s13 = sin(t13)
    s23 = sin(t23)
    return np.array([[c12*c13,
        c13*s12,
        s13/exp(1j*delta)],
        [-(c23*s12) - c12*exp(1j*delta)*s13*s23,
        c12*c23 - exp(1j*delta)*s12*s13*s23,
        c13*s23],
        [-(c12*c23*exp(1j*delta)*s13) + s12*s23,
        -(c23*exp(1j*delta)*s12*s13) - c12*s23,
        c13*c23]])