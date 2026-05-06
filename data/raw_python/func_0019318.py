def check_completeness(self):
        """Raise a |RuntimeError| if the |IOSequence.series| contains at
        least one |numpy.nan| value, if option |Options.checkseries| is
        enabled.

        >>> from hydpy import pub
        >>> pub.timegrids = '2000-01-01', '2000-01-11', '1d'
        >>> from hydpy.core.sequencetools import IOSequence
        >>> class Seq(IOSequence):
        ...     NDIM = 0
        >>> seq = Seq(None)
        >>> seq.activate_ram()
        >>> seq.check_completeness()
        Traceback (most recent call last):
        ...
        RuntimeError: The series array of sequence `seq` contains 10 nan values.

        >>> seq.series = 1.0
        >>> seq.check_completeness()
        
        >>> seq.series[3] = numpy.nan
        >>> seq.check_completeness()
        Traceback (most recent call last):
        ...
        RuntimeError: The series array of sequence `seq` contains 1 nan value.

        >>> with pub.options.checkseries(False):
        ...     seq.check_completeness()
        """
        if hydpy.pub.options.checkseries:
            isnan = numpy.isnan(self.series)
            if numpy.any(isnan):
                nmb = numpy.sum(isnan)
                valuestring = 'value' if nmb == 1 else 'values'
                raise RuntimeError(
                    f'The series array of sequence '
                    f'{objecttools.devicephrase(self)} contains '
                    f'{nmb} nan {valuestring}.')