def timeofyear(self):
        """Time of the year index (first simulation step of each year = 0...).

        The property |Indexer.timeofyear| is best explained through
        comparing it with property |Indexer.dayofyear|:

        Let us reconsider one of the examples of the documentation on
        property |Indexer.dayofyear|:

        >>> from hydpy import pub
        >>> from hydpy import Timegrids, Timegrid
        >>> from hydpy.core.indextools import Indexer
        >>> pub.timegrids = '27.02.2005', '3.03.2005', '1d'

        Due to the simulation stepsize being one day, the index arrays
        calculated by both properties are identical:

        >>> Indexer().dayofyear
        array([57, 58, 60, 61])
        >>> Indexer().timeofyear
        array([57, 58, 60, 61])

        In the next example the step size is halved:

        >>> pub.timegrids = '27.02.2005', '3.03.2005', '12h'

        Now the there a generally two subsequent simulation steps associated
        with the same day:

        >>> Indexer().dayofyear
        array([57, 57, 58, 58, 60, 60, 61, 61])

        However, the `timeofyear` array gives the index of the
        respective simulation steps of the actual year:

        >>> Indexer().timeofyear
        array([114, 115, 116, 117, 120, 121, 122, 123])

        Note the gap in the returned index array due to 2005 being not a
        leap year.
        """
        refgrid = timetools.Timegrid(
            timetools.Date('2000.01.01'),
            timetools.Date('2001.01.01'),
            hydpy.pub.timegrids.stepsize)

        def _timeofyear(date):
            date = copy.deepcopy(date)
            date.year = 2000
            return refgrid[date]

        return _timeofyear