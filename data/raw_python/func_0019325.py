def load_ext(self):
        """Read time series data like method |IOSequence.load_ext| of class
        |IOSequence|, but with special handling of missing data.

        When reading incomplete time series data, *HydPy* usually raises
        a |RuntimeError| to prevent from performing erroneous calculations.
        For instance, this makes sense for meteorological input data, being a
        definite requirement for hydrological simulations.  However, the
        same often does not hold for the time series of |Obs| sequences,
        e.g. representing measured discharge. Measured discharge is often
        handled as an optional input value, or even used for comparison
        purposes only.

        According to this reasoning, *HydPy* raises (at most) a |UserWarning|
        in case of missing or incomplete external time series data of |Obs|
        sequences.  The following examples show this based on the `LahnH`
        project, mainly focussing on the |Obs| sequence of node `dill`,
        which is ready for handling time series data at the end of the
        following steps:

        >>> from hydpy.core.examples import prepare_full_example_1
        >>> prepare_full_example_1()
        >>> from hydpy import HydPy, pub, TestIO
        >>> hp = HydPy('LahnH')
        >>> pub.timegrids = '1996-01-01', '1996-01-06', '1d'
        >>> with TestIO():
        ...     hp.prepare_network()
        ...     hp.init_models()
        ...     hp.prepare_obsseries()
        >>> obs = hp.nodes.dill.sequences.obs
        >>> obs.ramflag
        True

        Trying to read non-existing data raises the following warning
        and disables the sequence's ability to handle time series data:

        >>> with TestIO():
        ...     hp.load_obsseries()    # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        UserWarning: The `memory flag` of sequence `obs` of node `dill` had \
to be set to `False` due to the following problem: While trying to load the \
external data of sequence `obs` of node `dill`, the following error occurred: \
[Errno 2] No such file or directory: '...dill_obs_q.asc'
        >>> obs.ramflag
        False

        After writing a complete external data fine, everything works fine:

        >>> obs.activate_ram()
        >>> obs.series = 1.0
        >>> with TestIO():
        ...     obs.save_ext()
        >>> obs.series = 0.0
        >>> with TestIO():
        ...     obs.load_ext()
        >>> obs.series
        InfoArray([ 1.,  1.,  1.,  1.,  1.])

        Reading incomplete data also results in a warning message, but does
        not disable the |IOSequence.memoryflag|:

        >>> import numpy
        >>> obs.series[2] = numpy.nan
        >>> with TestIO():
        ...     pub.sequencemanager.nodeoverwrite = True
        ...     obs.save_ext()
        >>> with TestIO():
        ...     obs.load_ext()
        Traceback (most recent call last):
        ...
        UserWarning: While trying to load the external data of sequence `obs` \
of node `dill`, the following error occurred: The series array of sequence \
`obs` of node `dill` contains 1 nan value.
        >>> obs.memoryflag
        True

        Option |Options.warnmissingobsfile| allows disabling the warning
        messages without altering the functionalities described above:

        >>> hp.prepare_obsseries()
        >>> with TestIO():
        ...     with pub.options.warnmissingobsfile(False):
        ...         hp.load_obsseries()
        >>> obs.series
        InfoArray([  1.,   1.,  nan,   1.,   1.])
        >>> hp.nodes.lahn_1.sequences.obs.memoryflag
        False
        """
        try:
            super().load_ext()
        except OSError:
            del self.memoryflag
            if hydpy.pub.options.warnmissingobsfile:
                warnings.warn(
                    f'The `memory flag` of sequence '
                    f'{objecttools.nodephrase(self)} had to be set to `False` '
                    f'due to the following problem: {sys.exc_info()[1]}')
        except BaseException:
            if hydpy.pub.options.warnmissingobsfile:
                warnings.warn(str(sys.exc_info()[1]))