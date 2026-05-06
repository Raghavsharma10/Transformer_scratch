def ckm_wolfenstein(laC, A, rhobar, etabar):
    r"""CKM matrix in the Wolfenstein parametrization and standard phase
    convention.

    This function does not rely on an expansion in the Cabibbo angle but
    defines, to all orders in $\lambda$,

    - $\lambda = \sin\theta_{12}$
    - $A\lambda^2 = \sin\theta_{23}$
    - $A\lambda^3(\rho-i \eta) = \sin\theta_{13}e^{-i\delta}$

    where $\rho = \bar\rho/(1-\lambda^2/2)$ and
    $\eta = \bar\eta/(1-\lambda^2/2)$.

    Parameters
    ----------

    - `laC`: Wolfenstein parameter $\lambda$ (sine of Cabibbo angle)
    - `A`: Wolfenstein parameter $A$
    - `rhobar`: Wolfenstein parameter $\bar\rho = \rho(1-\lambda^2/2)$
    - `etabar`: Wolfenstein parameter $\bar\eta = \eta(1-\lambda^2/2)$
    """
    rho = rhobar/(1 - laC**2/2.)
    eta = etabar/(1 - laC**2/2.)
    return np.array([[sqrt(1 - laC**2)*sqrt(1 - A**2*laC**6*((-1j)*eta + rho)*((1j)*eta + rho)),
        laC*sqrt(1 - A**2*laC**6*((-1j)*eta + rho)*((1j)*eta + rho)),
        A*laC**3*((-1j)*eta + rho)],
        [-(laC*sqrt(1 - A**2*laC**4)) - A**2*laC**5*sqrt(1 - laC**2)*((1j)*eta + rho),
        sqrt(1 - laC**2)*sqrt(1 -  A**2*laC**4) - A**2*laC**6*((1j)*eta + rho),
        A*laC**2*sqrt(1 - A**2*laC**6*((-1j)*eta + rho)*((1j)*eta + rho))],
        [A*laC**3 - A*laC**3*sqrt(1 - laC**2)*sqrt(1 - A**2*laC**4)*((1j)*eta + rho),
        -(A*laC**2*sqrt(1 - laC**2)) - A*laC**4*sqrt(1 - A**2*laC**4)*((1j)*eta + rho),
        sqrt(1 - A**2*laC**4)*sqrt(1 - A**2*laC**6*((-1j)*eta + rho)*((1j)*eta + rho))]])