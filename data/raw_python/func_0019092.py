def update(self):
        """Calculate the smoothing parameter value.

        The following example is explained in some detail in module
        |smoothtools|:

        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> highestremotedischarge(1.0)
        >>> highestremotetolerance(0.0)
        >>> derived.highestremotesmoothpar.update()
        >>> from hydpy.cythons.smoothutils import smooth_min1
        >>> from hydpy import round_
        >>> round_(smooth_min1(-4.0, 1.5, derived.highestremotesmoothpar))
        -4.0
        >>> highestremotetolerance(2.5)
        >>> derived.highestremotesmoothpar.update()
        >>> round_(smooth_min1(-4.0, -1.5, derived.highestremotesmoothpar))
        -4.01

        Note that the example above corresponds to the example on function
        |calc_smoothpar_min1|, due to the value of parameter
        |HighestRemoteDischarge| being 1 m³/s.  Doubling the value of
        |HighestRemoteDischarge| also doubles the value of
        |HighestRemoteSmoothPar| proportional.  This leads to the following
        result:

        >>> highestremotedischarge(2.0)
        >>> derived.highestremotesmoothpar.update()
        >>> round_(smooth_min1(-4.0, 1.0, derived.highestremotesmoothpar))
        -4.02

        This relationship between |HighestRemoteDischarge| and
        |HighestRemoteSmoothPar| prevents from any smoothing when
        the value of |HighestRemoteDischarge| is zero:

        >>> highestremotedischarge(0.0)
        >>> derived.highestremotesmoothpar.update()
        >>> round_(smooth_min1(1.0, 1.0, derived.highestremotesmoothpar))
        1.0

        In addition, |HighestRemoteSmoothPar| is set to zero if
        |HighestRemoteDischarge| is infinity (because no actual value
        will ever come in the vicinit of infinity), which is why no
        value would be changed through smoothing anyway):

        >>> highestremotedischarge(inf)
        >>> derived.highestremotesmoothpar.update()
        >>> round_(smooth_min1(1.0, 1.0, derived.highestremotesmoothpar))
        1.0
        """
        control = self.subpars.pars.control
        if numpy.isinf(control.highestremotedischarge):
            self(0.0)
        else:
            self(control.highestremotedischarge *
                 smoothtools.calc_smoothpar_min1(control.highestremotetolerance)
                 )