def interp_w_v1(self):
    """Calculate the actual water stage based on linear interpolation.

    Required control parameters:
      |N|
      |llake_control.V|
      |llake_control.W|

    Required state sequence:
      |llake_states.V|

    Calculated state sequence:
      |llake_states.W|

    Examples:

        Prepare a model object:

        >>> from hydpy.models.llake import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')

        For the sake of brevity, define a test function:

        >>> def test(*vs):
        ...     for v in vs:
        ...         states.v.new = v
        ...         model.interp_w_v1()
        ...         print(repr(states.v), repr(states.w))

        Define a simple `w`-`v` relationship consisting of three nodes and
        calculate the water stages for different volumes:

        >>> n(3)
        >>> v(0., 2., 4.)
        >>> w(-1., 1., 2.)

        Perform the interpolation for a few test points:

        >>> test(0., .5, 2., 3., 4., 5.)
        v(0.0) w(-1.0)
        v(0.5) w(-0.5)
        v(2.0) w(1.0)
        v(3.0) w(1.5)
        v(4.0) w(2.0)
        v(5.0) w(2.5)

        The reference water stage of the relationship can be selected
        arbitrarily.  Even negative water stages are returned, as is
        demonstrated by the first two calculations.  For volumes outside
        the range of the (`v`,`w`) pairs, the outer two highest pairs are
        used for linear extrapolation.
    """
    con = self.parameters.control.fastaccess
    new = self.sequences.states.fastaccess_new
    for jdx in range(1, con.n):
        if con.v[jdx] >= new.v:
            break
    new.w = ((new.v-con.v[jdx-1]) *
             (con.w[jdx]-con.w[jdx-1]) /
             (con.v[jdx]-con.v[jdx-1]) +
             con.w[jdx-1])