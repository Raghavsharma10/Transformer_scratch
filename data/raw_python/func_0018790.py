def _calcidxs(func):
        """Return the required indexes based on the given lambda function
        and the |Timegrids| object handled by module |pub|.  Raise a
        |RuntimeError| if the latter is not available.
        """
        timegrids = hydpy.pub.get('timegrids')
        if timegrids is None:
            raise RuntimeError(
                'An Indexer object has been asked for an %s array.  Such an '
                'array has neither been determined yet nor can it be '
                'determined automatically at the moment.   Either define an '
                '%s array manually and pass it to the Indexer object, or make '
                'a proper Timegrids object available within the pub module.  '
                'In usual HydPy applications, the latter is done '
                'automatically.'
                % (func.__name__, func.__name__))
        idxs = numpy.empty(len(timegrids.init), dtype=int)
        for jdx, date in enumerate(hydpy.pub.timegrids.init):
            idxs[jdx] = func(date)
        return idxs