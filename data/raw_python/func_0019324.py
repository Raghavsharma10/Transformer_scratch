def load_ext(self):
        """Read time series data like method |IOSequence.load_ext| of class
        |IOSequence|, but with special handling of missing data.

        The method's "special handling" is to convert errors to warnings.
        We explain the reasons in the documentation on method |Obs.load_ext|
        of class |Obs|, from which we borrow the following examples.
        The only differences are that method |Sim.load_ext| of class |Sim|
        does not disable property |IOSequence.memoryflag| and uses option
        |Options.warnmissingsimfile| instead of |Options.warnmissingobsfile|:

        >>> from hydpy.core.examples import prepare_full_example_1
        >>> prepare_full_example_1()
        >>> from hydpy import HydPy, pub, TestIO
        >>> hp = HydPy('LahnH')
        >>> pub.timegrids = '1996-01-01', '1996-01-06', '1d'
        >>> with TestIO():
        ...     hp.prepare_network()
        ...     hp.init_models()
        ...     hp.prepare_simseries()
        >>> sim = hp.nodes.dill.sequences.sim
        >>> with TestIO():
        ...     sim.load_ext()    # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        UserWarning: While trying to load the external data of sequence \
`sim` of node `dill`, the following error occurred: [Errno 2] No such file \
or directory: '...dill_sim_q.asc'
        >>> sim.series
        InfoArray([ nan,  nan,  nan,  nan,  nan])

        >>> sim.series = 1.0
        >>> with TestIO():
        ...     sim.save_ext()
        >>> sim.series = 0.0
        >>> with TestIO():
        ...     sim.load_ext()
        >>> sim.series
        InfoArray([ 1.,  1.,  1.,  1.,  1.])

        >>> import numpy
        >>> sim.series[2] = numpy.nan
        >>> with TestIO():
        ...     pub.sequencemanager.nodeoverwrite = True
        ...     sim.save_ext()
        >>> with TestIO():
        ...     sim.load_ext()
        Traceback (most recent call last):
        ...
        UserWarning: While trying to load the external data of sequence `sim` \
of node `dill`, the following error occurred: The series array of sequence \
`sim` of node `dill` contains 1 nan value.
        >>> sim.series
        InfoArray([  1.,   1.,  nan,   1.,   1.])

        >>> sim.series = 0.0
        >>> with TestIO():
        ...     with pub.options.warnmissingsimfile(False):
        ...         sim.load_ext()
        >>> sim.series
        InfoArray([  1.,   1.,  nan,   1.,   1.])
        """
        try:
            super().load_ext()
        except BaseException:
            if hydpy.pub.options.warnmissingsimfile:
                warnings.warn(str(sys.exc_info()[1]))