def calculate_split_from_extents(self):
        """
        Returns split calculated using extents obtained
        from :attr:`split_begs` and :attr:`split_ends`.
        All calls to calculate the split end up here to produce
        the :mod:`numpy` `structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
        of :obj:`tuple`-of-:obj:`slice` elements.

        :rtype: :obj:`numpy.ndarray`
        :return:
           A :mod:`numpy` `structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
           where each element is a :obj:`tuple` of :obj:`slice` objects.
        """
        self.logger.debug("self.split_shape=%s", self.split_shape)
        self.logger.debug("self.split_begs=%s", self.split_begs)
        self.logger.debug("self.split_ends=%s", self.split_ends)

        ret = \
            _np.array(
                [
                    tuple(
                        [
                            slice(
                                max([
                                    self.split_begs[d][idx[d]]
                                    + self.array_start[d]
                                    - self.halo[d, 0]
                                    * (self.split_ends[d][idx[d]] > self.split_begs[d][idx[d]]),
                                    self.tile_beg_min[d]
                                ]),
                                min([
                                    self.split_ends[d][idx[d]]
                                    + self.array_start[d]
                                    + self.halo[d, 1]
                                    * (self.split_ends[d][idx[d]] > self.split_begs[d][idx[d]]),
                                    self.tile_end_max[d]
                                ])
                            )
                            for d in range(len(self.split_shape))
                        ]
                    )
                    for idx in
                    _np.array(
                        _np.unravel_index(
                            _np.arange(0, _np.product(self.split_shape)),
                            self.split_shape
                        )
                    ).T
                ],
                dtype=[("%d" % d, "object") for d in range(len(self.split_shape))]
            ).reshape(self.split_shape)

        return ret