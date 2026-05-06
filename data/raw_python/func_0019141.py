def simulationstep(timestep):
    """ Define a simulation time step size for testing purposes within a
    parameter control file.

    Using |simulationstep| only affects the values of time dependent
    parameters, when `pub.timegrids.stepsize` is not defined.  It thus has
    no influence on usual hydpy simulations at all.  Use it just to check
    your parameter control files.  Write it in a line immediately behind
    the one calling |parameterstep|.

    To clarify its purpose, executing raises a warning, when executing
    it from within a control file:

    >>> from hydpy import pub
    >>> with pub.options.warnsimulationstep(True):
    ...     from hydpy.models.hland_v1 import *
    ...     parameterstep('1d')
    ...     simulationstep('1h')
    Traceback (most recent call last):
    ...
    UserWarning: Note that the applied function `simulationstep` is intended \
for testing purposes only.  When doing a HydPy simulation, parameter values \
are initialised based on the actual simulation time step as defined under \
`pub.timegrids.stepsize` and the value given to `simulationstep` is ignored.
    >>> k4.simulationstep
    Period('1h')
    """
    if hydpy.pub.options.warnsimulationstep:
        warnings.warn(
            'Note that the applied function `simulationstep` is intended for '
            'testing purposes only.  When doing a HydPy simulation, parameter '
            'values are initialised based on the actual simulation time step '
            'as defined under `pub.timegrids.stepsize` and the value given '
            'to `simulationstep` is ignored.')
    parametertools.Parameter.simulationstep(timestep)