def update(self):
        """Calculate the smoothing parameter values.

        The following example is explained in some detail in module
        |smoothtools|:

        >>> from hydpy import pub
        >>> pub.timegrids = '2000.01.01', '2000.01.03', '1d'
        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> remotedischargesafety(0.0)
        >>> remotedischargesafety.values[1] = 2.5
        >>> derived.remotedischargesmoothpar.update()
        >>> from hydpy.cythons.smoothutils import smooth_logistic1
        >>> from hydpy import round_
        >>> round_(smooth_logistic1(0.1, derived.remotedischargesmoothpar[0]))
        1.0
        >>> round_(smooth_logistic1(2.5, derived.remotedischargesmoothpar[1]))
        0.99
        """
        metapar = self.subpars.pars.control.remotedischargesafety
        self.shape = metapar.shape
        self(tuple(smoothtools.calc_smoothpar_logistic1(mp)
                   for mp in metapar.values))