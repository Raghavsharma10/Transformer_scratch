def _best_res_pixels(self):
        """
        Returns a numpy array of all the HEALPix indexes contained in the MOC at its max order.

        Returns
        -------
        result : `~numpy.ndarray`
            The array of HEALPix at ``max_order``
        """
        factor = 2 * (AbstractMOC.HPY_MAX_NORDER - self.max_order)
        pix_l = []
        for iv in self._interval_set._intervals:
            for val in range(iv[0] >> factor, iv[1] >> factor):
                pix_l.append(val)

        return np.asarray(pix_l)