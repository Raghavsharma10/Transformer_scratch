def max_order(self):
        """
        Depth of the smallest HEALPix cells found in the MOC instance.
        """
        # TODO: cache value
        combo = int(0)
        for iv in self._interval_set._intervals:
            combo |= iv[0] | iv[1]

        ret = AbstractMOC.HPY_MAX_NORDER - (utils.number_trailing_zeros(combo) // 2)
        if ret < 0:
            ret = 0

        return ret