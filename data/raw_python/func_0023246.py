def from_nuniq_interval_set(cls, nuniq_is):
        """
        Convert an IntervalSet containing NUNIQ intervals to an IntervalSet representing HEALPix
        cells following the NESTED numbering scheme.

        Parameters
        ----------
        nuniq_is : `IntervalSet`
            IntervalSet object storing HEALPix cells as [ipix + 4*4^(order), ipix+1 + 4*4^(order)[ intervals.

        Returns
        -------
        interval : `IntervalSet`
            IntervalSet object storing HEALPix cells as [ipix*4^(29-order), (ipix+1)*4^(29-order)[ intervals.
        """
        nested_is = IntervalSet()
        # Appending a list is faster than appending a numpy array
        # For these algorithms we append a list and create the interval set from the finished list
        rtmp = []
        last_order = 0
        intervals = nuniq_is._intervals
        diff_order = IntervalSet.HPY_MAX_ORDER
        shift_order = 2 * diff_order
        for interval in intervals:
            for j in range(interval[0], interval[1]):
                order, i_pix = uniq2orderipix(j)

                if order != last_order:
                    nested_is = nested_is.union(IntervalSet(np.asarray(rtmp)))
                    rtmp = []
                    last_order = order
                    diff_order = IntervalSet.HPY_MAX_ORDER - order
                    shift_order = 2 * diff_order

                rtmp.append((i_pix << shift_order, (i_pix + 1) << shift_order))

        nested_is = nested_is.union(IntervalSet(np.asarray(rtmp)))
        return nested_is