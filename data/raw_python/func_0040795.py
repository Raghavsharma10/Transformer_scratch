def maxwellian(E: np.ndarray, E0: np.ndarray, Q0: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    input:
    ------
    E: 1-D vector of energy bins [eV]
    E0: characteristic energy (scalar or vector) [eV]
    Q0: flux coefficient (scalar or vector) (to yield overall flux Q)

    output:
    -------
    Phi: differential number flux
    Q: total flux

    Tanaka 2006 Eqn. 1
    http://odin.gi.alaska.edu/lumm/Papers/Tanaka_2006JA011744.pdf
    """
    E0 = np.atleast_1d(E0)
    Q0 = np.atleast_1d(Q0)
    assert E0.ndim == Q0.ndim == 1
    assert (Q0.size == 1 or Q0.size == E0.size)

    Phi = Q0/(2*pi*E0**3) * E[:, None] * np.exp(-E[:, None]/E0)

    Q = np.trapz(Phi, E, axis=0)
    logging.info('total maxwellian flux Q: ' + (' '.join('{:.1e}'.format(q) for q in Q)))
    return Phi, Q