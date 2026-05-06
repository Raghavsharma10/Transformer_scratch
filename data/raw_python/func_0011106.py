def maximum_violation(A_configuration, B_configuration, I, level, extra=None):
    """Get the maximum violation of a two-party Bell inequality.

    :param A_configuration: Measurement settings of Alice.
    :type A_configuration: list of int.
    :param B_configuration: Measurement settings of Bob.
    :type B_configuration: list of int.
    :param I: The I matrix of a Bell inequality in the Collins-Gisin notation.
    :type I: list of list of int.
    :param level: Level of relaxation.
    :type level: int.

    :returns: tuple of primal and dual solutions of the SDP relaxation.
    """
    P = Probability(A_configuration, B_configuration)
    objective = define_objective_with_I(I, P)
    if extra is None:
        extramonomials = []
    else:
        extramonomials = P.get_extra_monomials(extra)
    sdpRelaxation = SdpRelaxation(P.get_all_operators(), verbose=0)
    sdpRelaxation.get_relaxation(level, objective=objective,
                                 substitutions=P.substitutions,
                                 extramonomials=extramonomials)
    solve_sdp(sdpRelaxation)
    return sdpRelaxation.primal, sdpRelaxation.dual