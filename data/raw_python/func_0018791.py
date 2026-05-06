def dayofyear(self):
        """Day of the year index (the first of January = 0...).

        For reasons of consistency between leap years and non-leap years,
        assuming a daily time step, index 59 is always associated with the
        29th of February.  Hence, it is missing in non-leap years:

        >>> from hydpy import pub
        >>> from hydpy.core.indextools import Indexer
        >>> pub.timegrids = '27.02.2004', '3.03.2004', '1d'
        >>> Indexer().dayofyear
        array([57, 58, 59, 60, 61])
        >>> pub.timegrids = '27.02.2005', '3.03.2005', '1d'
        >>> Indexer().dayofyear
        array([57, 58, 60, 61])
        """
        def _dayofyear(date):
            return (date.dayofyear-1 +
                    ((date.month > 2) and (not date.leapyear)))
        return _dayofyear