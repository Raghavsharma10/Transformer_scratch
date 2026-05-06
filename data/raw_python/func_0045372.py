def ckm_tree(Vus, Vub, Vcb, gamma):
    r"""CKM matrix in the tree parametrization and standard phase
    convention.

    In this parametrization, the parameters are directly measured from
    tree-level $B$ decays. It is thus particularly suited for new physics
    analyses because the tree-level decays should be dominated by the Standard
    Model. This function involves no analytical approximations.

    Relation to the standard parametrization:

    - $V_{us} = \cos \theta_{13} \sin \theta_{12}$
    - $|V_{ub}| = |\sin \theta_{13}|$
    - $V_{cb} = \cos \theta_{13} \sin \theta_{23}$
    - $\gamma=\delta$

    Parameters
    ----------

    - `Vus`: CKM matrix element $V_{us}$
    - `Vub`: Absolute value of CKM matrix element $|V_{ub}|$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `gamma`: CKM phase $\gamma=\delta$ in radians
    """
    return np.array([[sqrt(1 - Vub**2)*sqrt(1 - Vus**2/(1 - Vub**2)),
        Vus,
        Vub/exp(1j*gamma)],
        [-((sqrt(1 - Vcb**2/(1 - Vub**2))*Vus)/sqrt(1 - Vub**2)) - (Vub*exp(1j*gamma)*Vcb*sqrt(1 - Vus**2/(1 - Vub**2)))/sqrt(1 - Vub**2),
        -((Vub*exp(1j*gamma)*Vcb*Vus)/(1 - Vub**2)) + sqrt(1 - Vcb**2/(1 - Vub**2))*sqrt(1 - Vus**2/(1 - Vub**2)),
        Vcb],
        [(Vcb*Vus)/(1 - Vub**2) - Vub*exp(1j*gamma)*sqrt(1 - Vcb**2/(1 - Vub**2))*sqrt(1 - Vus**2/(1 - Vub**2)),
        -((Vub*exp(1j*gamma)*sqrt(1 - Vcb**2/(1 - Vub**2))*Vus)/sqrt(1 - Vub**2)) - (Vcb*sqrt(1 - Vus**2/(1 - Vub**2)))/sqrt(1 - Vub**2),
        sqrt(1 - Vub**2)*sqrt(1 - Vcb**2/(1 - Vub**2))]])