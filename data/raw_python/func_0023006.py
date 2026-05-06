def _uniq_pixels_iterator(self):
        """
        Generator giving the NUNIQ HEALPix pixels of the MOC.

        Returns
        -------
        uniq :
            the NUNIQ HEALPix pixels iterator
        """
        intervals_uniq_l = IntervalSet.to_nuniq_interval_set(self._interval_set)._intervals
        for uniq_iv in intervals_uniq_l:
            for uniq in range(uniq_iv[0], uniq_iv[1]):
                yield uniq