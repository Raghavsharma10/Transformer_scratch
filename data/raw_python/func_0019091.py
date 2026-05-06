def update(self):
        """Calculate the smoothing parameter value.

        The following example is explained in some detail in module
        |smoothtools|:

        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> waterlevelminimumremotetolerance(0.0)
        >>> derived.waterlevelminimumremotesmoothpar.update()
        >>> from hydpy.cythons.smoothutils import smooth_logistic1
        >>> from hydpy import round_
        >>> round_(smooth_logistic1(0.1,
        ...        derived.waterlevelminimumremotesmoothpar))
        1.0
        >>> waterlevelminimumremotetolerance(2.5)
        >>> derived.waterlevelminimumremotesmoothpar.update()
        >>> round_(smooth_logistic1(2.5,
        ...        derived.waterlevelminimumremotesmoothpar))
        0.99
        """
        metapar = self.subpars.pars.control.waterlevelminimumremotetolerance
        self(smoothtools.calc_smoothpar_logistic1(metapar))