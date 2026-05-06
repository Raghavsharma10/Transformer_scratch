def mea_approximation(model, max_order, closure='scalar', *closure_args, **closure_kwargs):
    r"""
    A wrapper around :class:`~means.approximation.mea.moment_expansion_approximation.MomentExpansionApproximation`.
    It performs moment expansion approximation (MEA) up to a given order of moment.
    See :class:`~means.approximation.mea.moment_expansion_approximation.MomentExpansionApproximation` for details
    about the options.


    :return: an ODE problem which can be further used in inference and simulation.
    :rtype: :class:`~means.core.problems.ODEProblem`
    """
    mea = MomentExpansionApproximation(model, max_order, closure=closure, *closure_args, **closure_kwargs)
    return mea.run()