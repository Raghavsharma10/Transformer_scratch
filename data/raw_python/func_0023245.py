def to_nuniq_interval_set(cls, nested_is):
        """
        Convert an IntervalSet using the NESTED numbering scheme to an IntervalSet containing UNIQ numbers for HEALPix
        cells.

        Parameters
        ----------
        nested_is : `IntervalSet`
            IntervalSet object storing HEALPix cells as [ipix*4^(29-order), (ipix+1)*4^(29-order)[ intervals.

        Returns
        -------
        interval : `IntervalSet`
            IntervalSet object storing HEALPix cells as [ipix + 4*4^(order), ipix+1 + 4*4^(order)[ intervals.
        """
        r2 = nested_is.copy()
        res = []

        if r2.empty():
            return IntervalSet()

        order = 0
        while not r2.empty():
            shift = int(2 * (IntervalSet.HPY_MAX_ORDER - order))
            ofs = (int(1) << shift) - 1
            ofs2 = int(1) << (2 * order + 2)

            r4 = []
            for iv in r2._intervals:
                a = (int(iv[0]) + ofs) >> shift
                b = int(iv[1]) >> shift

                c = a << shift
                d = b << shift
                if d > c:
                    r4.append((c, d))
                    res.append((a + ofs2, b + ofs2))

            if len(r4) > 0:
                r4_is = IntervalSet(np.asarray(r4))
                r2 = r2.difference(r4_is)

            order += 1

        return IntervalSet(np.asarray(res))